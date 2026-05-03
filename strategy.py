#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves; alignment with weekly trend
# and volume confirmation increases reliability. Designed for very low trade frequency
# (7-25/year) on 1d timeframe to minimize fee drag. Works in both bull and bear markets
# by trading breakouts in the direction of the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Donchian channels and volume EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter (on weekly data later)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 1d timeframe (trivial alignment but required for consistency)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate Donchian channels (20-period) from 1d data
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(df_1d['high'].values)
    low_series = pd.Series(df_1d['low'].values)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction using 1w EMA50
        df_1w = get_htf_data(prices, '1w')
        if len(df_1w) > 0:
            ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
            ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
            is_uptrend = close[i] > ema_50_1w_aligned[i]
            is_downtrend = close[i] < ema_50_1w_aligned[i]
        else:
            is_uptrend = True
            is_downtrend = True
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike in uptrend
            if close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and is_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike in downtrend
            elif close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and is_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters below upper Donchian
            if close[i] < donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters above lower Donchian
            if close[i] > donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals