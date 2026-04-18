#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions; EMA filter ensures trades align with higher timeframe trend.
# Volume confirmation filters out low-momentum breakouts. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 12-37 trades/year (50-150 total over 4 years) to avoid fee drag.
name = "6h_WilliamsR_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(50)
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Align 1d EMA to 6h (wait for daily close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wr = williams_r[i]
        ema_val = ema_50_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price above EMA + volume
            if wr < -80 and close_val > ema_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price below EMA + volume
            elif wr > -20 and close_val < ema_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or price below EMA
            if wr > -20 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or price above EMA
            if wr < -80 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals