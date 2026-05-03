#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channels provide robust price structure with clear breakout levels.
# Trend filter ensures we trade with the higher timeframe momentum to avoid counter-trend whipsaws.
# Volume confirmation filters low-conviction breakouts. Designed for 15-25 trades/year on 4h to minimize fee drag.
# Works in bull markets via buying upper band breakouts in uptrends and bear markets via selling lower band breakdowns in downtrends.

name = "4h_Donchian20_1dEMA50_VolumeConfirm"
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        upper_band[i] = np.max(high[i-lookback+1:i+1])
        lower_band[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 4h volume EMA (20-period) for confirmation
    vol_ema_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ema_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period EMA
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        if position == 0:
            # Long: bullish breakout above upper band in 1d uptrend with volume spike
            if breakout_up and ema_50_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below lower band in 1d downtrend with volume spike
            elif breakout_down and ema_50_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to upper band or loses 1d uptrend
            if close[i] < upper_band[i] or ema_50_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to lower band or loses 1d downtrend
            if close[i] > lower_band[i] or ema_50_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals