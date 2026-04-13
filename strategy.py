# 12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation
# Hypothesis: 12h timeframe with 1d context for trend filtering.
# Uses 1d high/low to determine trend direction and 12h Camarilla pivots for entry.
# Volume confirmation on breakout. Works in bull/bear by trading with 1d trend.
# Target: 12-37 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 12h data for entries
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 1d trend: 50 EMA
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Camarilla pivots (R4/S4 breakout with volume)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    close_prev_12h = np.roll(close_12h, 1)
    close_prev_12h[0] = close_12h[0]
    
    range_12h = high_12h - low_12h
    
    # Resistance levels (focus on R4 for breakout)
    R4_12h = close_prev_12h + (range_12h * 1.5000 / 2)
    # Support levels (focus on S4 for breakdown)
    S4_12h = close_prev_12h - (range_12h * 1.5000 / 2)
    
    # Align 12h levels to minute timeframe
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, R4_12h)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, S4_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_12h_aligned[i]) or np.isnan(S4_12h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend direction
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Long breakout: price breaks above R4 with volume expansion in uptrend
        long_breakout = close[i] > R4_12h_aligned[i] and volume_expansion[i] and uptrend
        
        # Short breakdown: price breaks below S4 with volume expansion in downtrend
        short_breakout = close[i] < S4_12h_aligned[i] and volume_expansion[i] and downtrend
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "12h_1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0