#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and session filter.
# Uses 4h Donchian channels (20-period) for trend direction and breakout signals.
# Volume spike (>1.5x 20-period EMA) confirms institutional participation.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drag on 1h timeframe.
# Works in both bull and bear markets by trading with the 4h trend and using volume as confirmation.

name = "1h_Donchian20_4hVolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) using previous period's data to avoid look-ahead
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # Donchian upper = max(high, lookback=20), lower = min(low, lookback=20)
    donch_h = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume spike (volume > 1.5 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_4h['volume'].values > (1.5 * vol_ema_20)
    
    # Align 4h indicators to 1h timeframe
    donch_h_aligned = align_htf_to_ltf(prices, df_4h, donch_h)
    donch_l_aligned = align_htf_to_ltf(prices, df_4h, donch_l)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donch_h_aligned[i]) or np.isnan(donch_l_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian high with volume spike
            if high[i] > donch_h_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian low with volume spike
            elif low[i] < donch_l_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below 4h Donchian low (reversal)
            if low[i] < donch_l_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above 4h Donchian high (reversal)
            if high[i] > donch_h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals