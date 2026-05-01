#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Donchian(20) breakout + 1d volume spike + session filter (08-20 UTC)
# Uses 4h for signal direction and structure, 1h only for precise entry timing
# Volume spike confirms breakout legitimacy, session filter reduces noise trades
# Discrete sizing (0.20) and tight conditions target 60-150 total trades over 4 years
# Works in bull/bear: 4h Donchian provides clear structure, volume confirms institutional participation

name = "1h_Donchian20_1dVolume_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h HTF data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d HTF data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period)
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe (waits for completed 4h bar)
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # 1d volume spike filter: volume > 1.5 * 20-period EMA
    vol_1d = df_1d['volume'].values
    vol_ema_20_1d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ema_20_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Donchian (20 bars)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if not in trading session or missing data
        if not in_session[i] or np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above 4h Donchian high with 1d volume spike
            if close[i] > highest_high_aligned[i] and volume_spike_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # Short: Break below 4h Donchian low with 1d volume spike
            elif close[i] < lowest_low_aligned[i] and volume_spike_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price returns to 4h Donchian low
            if close[i] <= lowest_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price returns to 4h Donchian high
            if close[i] >= highest_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals