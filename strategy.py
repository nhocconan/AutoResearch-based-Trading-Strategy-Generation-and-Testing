#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_R2S2_Breakout_V1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Points (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Set first values to avoid look-ahead
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Classic pivot (same for Camarilla)
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla R2 and S2 levels (stronger breakout levels)
    r2 = pivot + (range_val * 1.1 / 6)
    s2 = pivot - (range_val * 1.1 / 6)
    
    # Align to 12h timeframe
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # === Volume Confirmation (12h) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma10 = vol_series.rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / np.where(vol_ma10 > 0, vol_ma10, np.nan)
    
    # === Momentum Filter: 12h RSI > 50 for long, < 50 for short ===
    close_series = pd.Series(prices['close'].values)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        pivot_val = pivot_aligned[i]
        rsi_val = rsi_values[i]
        
        # Skip if any value is NaN
        if (np.isnan(vol_ratio_val) or np.isnan(r2_val) or 
            np.isnan(s2_val) or np.isnan(pivot_val) or 
            np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R2 with volume confirmation and bullish momentum
            if close_val > r2_val and vol_ratio_val > 1.5 and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: Break below S2 with volume confirmation and bearish momentum
            elif close_val < s2_val and vol_ratio_val > 1.5 and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot OR momentum turns bearish
            if close_val < pivot_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot OR momentum turns bullish
            if close_val > pivot_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals