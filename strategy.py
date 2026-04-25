#!/usr/bin/env python3
"""
6h Weekly Pivot Range Breakout + 1d ATR Filter + Volume Spike
Hypothesis: Weekly pivot ranges (R1/S1, R2/S2, R3/S3) act as key support/resistance zones.
Breakouts beyond R3 or below S3 with volume confirmation and filtered by 1d ATR (low volatility = range, high volatility = breakout)
capture strong momentum moves. Uses discrete sizing (0.0, ±0.25) to minimize fees.
Designed for 6h timeframe with tight entries to achieve 12-37 trades/year.
Works in both bull (breakouts up) and bear (breakdowns down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (based on prior week)
    prev_week_high = df_1w['high'].shift(1).values
    prev_week_low = df_1w['low'].shift(1).values
    prev_week_close = df_1w['close'].shift(1).values
    
    weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    weekly_range = prev_week_high - prev_week_low
    
    # Camarilla-style weekly levels (H3/L3 = strong S/R)
    h3_weekly = weekly_pivot + weekly_range * 1.1 / 4.0  # Resistance 3
    l3_weekly = weekly_pivot - weekly_range * 1.1 / 4.0  # Support 3
    
    # Align weekly levels to 6h
    h3_weekly_aligned = align_htf_to_ltf(prices, df_1w, h3_weekly)
    l3_weekly_aligned = align_htf_to_ltf(prices, df_1w, l3_weekly)
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no prior close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike: current 6h volume > 2.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for weekly data, ATR, and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(h3_weekly_aligned[i]) or np.isnan(l3_weekly_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        atr_val = atr_1d_aligned[i]
        vol_spike = volume_spike[i]
        h3_level = h3_weekly_aligned[i]
        l3_level = l3_weekly_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above weekly H3 AND volume spike AND ATR > 0 (always true, but keeps structure)
            long_entry = (curr_high > h3_level) and vol_spike
            # Short: price breaks below weekly L3 AND volume spike
            short_entry = (curr_low < l3_level) and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses back below weekly H3 (failed breakout) OR ATR drops below 50% of recent average (low vol = range)
            # Use simple ATR mean reversion exit: if ATR < 0.6 * 20-period ATR mean, exit (assume range returning)
            if i >= 20:
                atr_ma_20 = np.nanmean(atr_1d_aligned[max(0, i-19):i+1])
                if not np.isnan(atr_ma_20) and atr_val < 0.6 * atr_ma_20:
                    signals[i] = 0.0
                    position = 0
                elif curr_low < h3_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses back above weekly L3 OR ATR drops below 50% of recent average
            if i >= 20:
                atr_ma_20 = np.nanmean(atr_1d_aligned[max(0, i-19):i+1])
                if not np.isnan(atr_ma_20) and atr_val < 0.6 * atr_ma_20:
                    signals[i] = 0.0
                    position = 0
                elif curr_high > l3_level:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_H3L3_Breakout_1dATRFilter_VolumeSpike"
timeframe = "6h"
leverage = 1.0