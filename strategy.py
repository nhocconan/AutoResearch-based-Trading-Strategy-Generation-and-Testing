#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Trend
Hypothesis: Uses 12h timeframe with 1d Camarilla pivot levels for breakout trading.
Trades breakouts above/below pivot support/resistance levels with volume confirmation.
Follows 1d trend direction to avoid counter-trend whipsaws. Designed for 15-30 trades/year.
Works in bull/bear by following higher timeframe trend - avoids counter-trend losses.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # We'll use H3/L3 as resistance/support and H4/L4 as breakout levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate typical price for pivot point
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    # H4 = close + 1.5 * range
    # L4 = close - 1.5 * range
    # H3 = close + 1.25 * range
    # L3 = close - 1.25 * range
    H4 = close_1d + 1.5 * range_1d
    L4 = close_1d - 1.5 * range_1d
    H3 = close_1d + 1.25 * range_1d
    L3 = close_1d - 1.25 * range_1d
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_H4 = close[i] > H4_aligned[i-1]  # Break above H4 level
        breakdown_L4 = close[i] < L4_aligned[i-1]  # Break below L4 level
        
        # Entry conditions: only trade in direction of 1d trend
        long_entry = breakout_H4 and volume_filter and uptrend
        short_entry = breakdown_L4 and volume_filter and downtrend
        
        # Exit conditions: return to opposite H3/L3 level or trend reversal
        long_exit = (close[i] < L3_aligned[i]) or (not uptrend)  # Break below L3 or trend change
        short_exit = (close[i] > H3_aligned[i]) or (not downtrend)  # Break above H3 or trend change
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals