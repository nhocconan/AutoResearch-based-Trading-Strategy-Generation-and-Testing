#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R + 1d EMA trend + volume confirmation.
# Williams %R(14) identifies overbought/oversold conditions for mean reversion.
# 1d EMA50 filters trend direction (long only above, short only below).
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in ranging markets via mean reversion and in trending markets via trend alignment.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_WilliamsR14_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Williams %R
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Williams %R on 4h data: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    close_4h = pd.Series(df_4h['close'].values)
    
    highest_high = high_4h.rolling(window=14, min_periods=14).max()
    lowest_low = low_4h.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values  # Handle division by zero
    
    # Align Williams %R to lower timeframe (4h)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d data
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA to lower timeframe (4h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        wr_val = williams_r_aligned[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND price above EMA50 AND volume spike
            if wr_val < -80 and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND price below EMA50 AND volume spike
            elif wr_val > -20 and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R overbought (> -20) or price below EMA50
            if wr_val > -20 or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R oversold (< -80) or price above EMA50
            if wr_val < -80 or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals