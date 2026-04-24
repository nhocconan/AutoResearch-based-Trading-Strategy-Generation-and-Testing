#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for Donchian calculation, 1w for pivot-based trend filter (price above/below weekly pivot defines bull/bear regime).
- Entry: Long when price breaks above 6h Donchian upper in bull regime with volume > 1.5 * 6h volume MA(20);
         Short when price breaks below 6h Donchian lower in bear regime with volume > 1.5 * 6h volume MA(20).
- Exit: ATR trailing stop (2.0 * ATR(14)) or opposite Donchian breakout.
- Signal size: 0.25 discrete to balance capture and fee control.
- Weekly pivot: P = (weekly_high + weekly_low + weekly_close)/3; bull if close > P, bear if close < P.
- Works in bull (breakouts with trend) and bear (strong moves after panic lows/highs).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate weekly pivot for trend filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Calculate 6h volume MA(20) for confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    # Calculate 6h ATR(14) for trailing stop
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_6h_aligned = align_htf_to_ltf(prices, df_6h, atr_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20, 14, 1)  # Donchian needs 20, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma_6h_aligned[i]) or np.isnan(atr_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike confirmation: 1.5x threshold (balanced to avoid overtrading)
        vol_spike = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Trend filter: price above/below weekly pivot
        bull_regime = curr_close > weekly_pivot_aligned[i]
        bear_regime = curr_close < weekly_pivot_aligned[i]
        
        if position == 0:
            # Check for entry signals
            # Long: price breaks above 1d Donchian upper in bull regime with volume spike
            if curr_close > donchian_high_aligned[i] and bull_regime and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
            # Short: price breaks below 1d Donchian lower in bear regime with volume spike
            elif curr_close < donchian_low_aligned[i] and bear_regime and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = curr_low
        elif position == 1:
            # Long position: update highest and check exit conditions
            highest_since_entry = max(highest_since_entry, curr_high)
            # Exit: ATR trailing stop or opposite breakout (below Donchian low)
            if curr_low <= highest_since_entry - 2.0 * atr_6h_aligned[i] or curr_close < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: update lowest and check exit conditions
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # Exit: ATR trailing stop or opposite breakout (above Donchian high)
            if curr_high >= lowest_since_entry + 2.0 * atr_6h_aligned[i] or curr_close > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wPivot_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0