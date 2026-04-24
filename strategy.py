#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d volume regime filter and ATR-based stops.
- Primary timeframe: 4h for execution, HTF: 1d for volume regime (high/low volume days).
- Volume regime: 1d volume > 1.5 * 20-day volume MA = high conviction day (trend likely to continue).
                 1d volume < 0.5 * 20-day volume MA = low conviction day (avoid new entries).
- Entry: Long when price closes above Donchian(20) upper AND 1d volume regime = high.
         Short when price closes below Donchian(20) lower AND 1d volume regime = high.
- Exit: Opposite Donchian breakout or 1d volume regime shifts to low.
- Volume confirmation on 4h: current 4h volume > 1.2 * 20-period 4h volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume regime: ratio of current volume to 20-day volume MA
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / (vol_ma_1d + 1e-10)  # Avoid division by zero
    
    # Align 1d volume ratio to 4h
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation on 4h: current volume > 1.2 * 20-period volume MA
    volume_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.2 * volume_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1d bars for volume MA and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ratio_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = vol_ratio_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals - only in high volume conviction days
            if vol_ratio > 1.5 and volume_spike_4h[i]:  # High conviction day + 4h volume spike
                # Bullish breakout: price closes above upper Donchian
                if curr_close > highest_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian
                elif curr_close < lowest_low[i]:
                    signals[i] = -0.25
                    position = -1
            # Optional: mean reversion in low conviction days (commented out to reduce trades)
            # elif vol_ratio < 0.5:  # Low conviction day
            #     # Long when price touches lower Donchian and shows reversal
            #     if curr_low <= lowest_low[i] and curr_close > curr_low:
            #         signals[i] = 0.15
            #         position = 1
            #     # Short when price touches upper Donchian and shows reversal
            #     elif curr_high >= highest_high[i] and curr_close < curr_high:
            #         signals[i] = -0.15
            #         position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR volume regime drops to low conviction
            if curr_close < donchian_mid[i] or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR volume regime drops to low conviction
            if curr_close > donchian_mid[i] or vol_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0