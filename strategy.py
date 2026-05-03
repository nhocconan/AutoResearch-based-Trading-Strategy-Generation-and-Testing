#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1w EMA50 ensures alignment with weekly trend
# Volume confirmation filters false breakouts. Designed for low trade frequency (12-37/year) on 12h timeframe.
# Works in both bull and bear markets by only trading breakouts in the direction of the weekly trend.

name = "12h_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "12h"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume spike (volume > 1.8 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (1.8 * vol_ema_20)
    
    # Align 1w indicators to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    # Calculate Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper channel in uptrend with volume spike
            if close[i] > highest_high[i-1] and is_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian lower channel in downtrend with volume spike
            elif close[i] < lowest_low[i-1] and is_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian middle (mean reversion) or reverses from overbought
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < mid_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price closes above Donchian middle (mean reversion) or reverses from oversold
            mid_channel = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > mid_channel:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals