#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike
Hypothesis: Donchian(20) breakout on 1d timeframe with 1w EMA50 trend filter and volume spike confirmation.
Only trade breakouts in direction of weekly trend when volume exceeds 1.5x 20-day average volume.
Uses discrete position sizing (0.25) to minimize fee churn. Target trade frequency: ~10-20/year.
Designed to work in both bull and bear markets via trend alignment and volume confirmation.
Donchian breakouts represent strong momentum shifts with lower false signals when combined with
weekly trend filter and volume spike, reducing whipsaw in choppy markets.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 1d timeframe (standard 1-bar delay for EMA)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate Donchian channels on 1d data (20-period)
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume spike filter
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = avg_volume_20 * 1.5  # 1.5x average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20) and EMA50 (50)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(avg_volume_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for Donchian breakout signals with trend and volume filters
            # Long: price breaks above upper Donchian band in uptrend (close > EMA50) with volume spike
            # Short: price breaks below lower Donchian band in downtrend (close < EMA50) with volume spike
            long_signal = (close[i] > highest_high[i]) and (close[i] > ema50_aligned[i]) and (volume[i] > volume_spike_threshold[i])
            short_signal = (close[i] < lowest_low[i]) and (close[i] < ema50_aligned[i]) and (volume[i] > volume_spike_threshold[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below EMA50 (trend reversal)
            exit_signal = close[i] < ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above EMA50 (trend reversal)
            exit_signal = close[i] > ema50_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike"
timeframe = "1d"
leverage = 1.0