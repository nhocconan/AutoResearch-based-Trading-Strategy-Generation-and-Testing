#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WilliamsR_MeanReversion"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d: Williams %R (14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) != 0, williams_r, -50)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average) with min_periods
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # === 6h: RSI(14) for momentum confirmation ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for Williams %R and RSI
        # Get values
        close_val = close[i]
        williams_r_val = williams_r_aligned[i]
        vol_ratio_val = vol_ratio[i]
        rsi_val = rsi[i]
        
        # Skip if any value is NaN
        if np.isnan(williams_r_val) or np.isnan(vol_ratio_val) or np.isnan(rsi_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + RSI < 40 (weak momentum) + volume confirmation
            if williams_r_val < -80 and rsi_val < 40 and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + RSI > 60 (strong momentum) + volume confirmation
            elif williams_r_val > -20 and rsi_val > 60 and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or RSI > 60
            if williams_r_val > -50 or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or RSI < 40
            if williams_r_val < -50 or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals