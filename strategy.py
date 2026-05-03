#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Bull power = High - EMA13, Bear power = Low - EMA13.
# Long when bull power > 0 and increasing in 1d uptrend with volume spike.
# Short when bear power < 0 and decreasing in 1d downtrend with volume spike.
# Uses 6h timeframe to reduce trade frequency vs lower TFs. Designed for 12-37 trades/year to minimize fee drag.
# Works in bull markets via buying strength and in bear markets via selling weakness.

name = "6h_ElderRay_Trend_VolumeSpike"
timeframe = "6h"
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
    
    # Get 1d data for trend filter and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray and trend filter
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d EMA34 for trend filter (direction)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Precompute 6h EMA13 for Elder Ray calculation
    close_s = pd.Series(close)
    ema_13_6h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after sufficient warmup for EMA13
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_13_6h[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Elder Ray components using 6h data
        bull_power = high[i] - ema_13_6h[i]   # Bull power: High - EMA13
        bear_power = low[i] - ema_13_6h[i]    # Bear power: Low - EMA13
        
        # Volume confirmation: 20-period EMA of 6h volume
        vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Trend filter: 1d EMA34 direction
        trend_up = ema_34_1d_aligned[i] > close[i]
        trend_down = ema_34_1d_aligned[i] < close[i]
        
        if position == 0:
            # Long: bull power positive AND increasing (momentum) in 1d uptrend with volume spike
            if bull_power > 0 and bull_power > (high[i-1] - ema_13_6h[i-1]) and trend_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bear power negative AND decreasing (momentum) in 1d downtrend with volume spike
            elif bear_power < 0 and bear_power < (low[i-1] - ema_13_6h[i-1]) and trend_down and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bull power turns negative or loses 1d uptrend
            if bull_power <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bear power turns positive or loses 1d downtrend
            if bear_power >= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals