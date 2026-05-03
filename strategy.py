#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation
# Donchian channels capture volatility-based breakouts. In bull markets, price > EMA34 (uptrend) favors long breakouts.
# In bear markets, price < EMA34 (downtrend) favors short breakouts. Volume spike confirms conviction.
# Designed for 20-50 trades/year on 4h to minimize fee drag while maintaining edge in all regimes.

name = "4h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Precompute 20-period volume EMA for efficiency
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Donchian and EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels using data up to current bar
        highest_high = np.max(high[i-19:i+1])
        lowest_low = np.min(low[i-19:i+1])
        
        # Volume spike condition: current volume > 2.0 * 20-period volume EMA
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        breakout_long = close[i] > highest_high and volume_spike
        breakout_short = close[i] < lowest_low and volume_spike
        
        if position == 0:
            # Long: break above upper channel in 1d uptrend with volume spike
            if breakout_long and ema_34_1d_aligned[i] < close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel in 1d downtrend with volume spike
            elif breakout_short and ema_34_1d_aligned[i] > close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper channel or loses 1d uptrend
            if close[i] < highest_high or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower channel or loses 1d downtrend
            if close[i] > lowest_low or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals