#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels provide clear structure for breakouts in both bull and bear markets.
# 1w EMA50 ensures alignment with the weekly trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts and confirms institutional participation.
# Designed for low trade frequency (target: 7-25/year) to minimize fee drag on 1d timeframe.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend with volume).

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA trend filter and volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1w volume confirmation (volume > 1.5 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmed = df_1w['volume'].values > (1.5 * vol_ema_20)
    volume_confirmed_aligned = align_htf_to_ltf(prices, df_1w, volume_confirmed)
    
    # Calculate 1d Donchian channels (20-period)
    # Use rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_confirmed_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume confirmation
            # In uptrend: follow the trend
            # In downtrend: require stronger volume confirmation to avoid fakeouts
            if high[i] > donchian_high[i] and volume_confirmed_aligned[i]:
                if is_uptrend or (not is_uptrend and volume_confirmed_aligned[i]):
                    signals[i] = 0.25
                    position = 1
            # Short: Price breaks below Donchian low with volume confirmation
            # In downtrend: follow the trend
            # In uptrend: require stronger volume confirmation to avoid fakeouts
            elif low[i] < donchian_low[i] and volume_confirmed_aligned[i]:
                if is_downtrend or (not is_downtrend and volume_confirmed_aligned[i]):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: Price breaks below Donchian low (reversal) or time-based exit
            if low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above Donchian high (reversal) or time-based exit
            if high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals