#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with 1h market structure filter.
Trade 1d Donchian channel breakouts with 1h market structure (HH/HL vs LH/LL) and volume confirmation.
Use 1d for signal generation and 1h for trend confirmation to avoid false breakouts in choppy markets.
Targets 15-25 trades/year by requiring confluence of daily breakout, hourly structure, and volume.
Works in bull markets via trend-following breakouts and in bear via avoiding false breakouts during downtrends.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels (main signal)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1h data for market structure (trend filter)
    df_1h = get_htf_data(prices, '1h')
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate 1h swing points: Higher Highs (HH) and Higher Lows (HL) for uptrend
    # Lower Highs (LH) and Lower Lows (LL) for downtrend
    # Simple approach: compare current high/low to previous swing points
    # We'll use a 5-period lookback for swing identification
    hh = np.zeros_like(high_1h, dtype=bool)  # Higher High
    hl = np.zeros_like(low_1h, dtype=bool)   # Higher Low
    lh = np.zeros_like(high_1h, dtype=bool)  # Lower High
    ll = np.zeros_like(low_1h, dtype=bool)   # Lower Low
    
    # Calculate swing points
    for i in range(5, len(high_1h)):
        # Higher High: current high > previous high and previous high > high before that
        if high_1h[i] > high_1h[i-1] and high_1h[i-1] > high_1h[i-2]:
            hh[i] = True
        # Higher Low: current low > previous low and previous low > low before that
        if low_1h[i] > low_1h[i-1] and low_1h[i-1] > low_1h[i-2]:
            hl[i] = True
        # Lower High: current high < previous high and previous high < high before that
        if high_1h[i] < high_1h[i-1] and high_1h[i-1] < high_1h[i-2]:
            lh[i] = True
        # Lower Low: current low < previous low and previous low < low before that
        if low_1h[i] < low_1h[i-1] and low_1h[i-1] < low_1h[i-2]:
            ll[i] = True
    
    # Determine trend: uptrend if HH and HL, downtrend if LH and LL
    uptrend = hh & hl
    downtrend = lh & ll
    
    # Align 1d and 1h data to 1d timeframe (since we're generating signals on 1d)
    high_max_20_aligned = align_htf_to_ltf(df_1d, df_1d, high_max_20)  # 1d to 1d (no change)
    low_min_20_aligned = align_htf_to_ltf(df_1d, df_1d, low_min_20)   # 1d to 1d (no change)
    
    # For 1h data aligned to 1d: we need the 1h trend value that corresponds to each 1d bar
    # Since 1d = 24 * 1h bars, we take the last 1h value of each day
    uptrend_aligned = align_htf_to_ltf(df_1d, df_1h, uptrend.astype(float))
    downtrend_aligned = align_htf_to_ltf(df_1d, df_1h, downtrend.astype(float))
    
    # Volume filter: current volume > 1.3x 20-period average (to avoid noise)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need at least 20 days for Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20_aligned[i]) or np.isnan(low_min_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume and 1h uptrend
            if close[i] > high_max_20_aligned[i] and volume_filter[i] and uptrend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume and 1h downtrend
            elif close[i] < low_min_20_aligned[i] and volume_filter[i] and downtrend_aligned[i] > 0.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1d Donchian low (mean reversion) OR trend breaks down
            if close[i] < low_min_20_aligned[i] or downtrend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1d Donchian high (mean reversion) OR trend breaks up
            if close[i] > high_max_20_aligned[i] or uptrend_aligned[i] > 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1hStructure_Volume_Filter"
timeframe = "1d"
leverage = 1.0