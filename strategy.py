#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via aligned SMAs.
# In bull: JAW > TEETH > LIPS (green alignment) → long bias
# In bear: JAW < TEETH < LIPS (red alignment) → short bias
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Target: 50-150 total trades over 4 years.

name = "6h_WilliamsAlligator_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # JAW: 13-period SMMA, TEETH: 8-period SMMA, LIPS: 5-period SMMA
    def smma(source, period):
        # Smoothed Moving Average: first value is SMA, then recursive
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return result
        # First value: SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start from JAW period to have valid Alligator
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA on 6h
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: Alligator aligned bullish (JAW > TEETH > LIPS) in 1d uptrend with volume spike
            if jaw[i] > teeth[i] and teeth[i] > lips[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned bearish (JAW < TEETH < LIPS) in 1d downtrend with volume spike
            elif jaw[i] < teeth[i] and teeth[i] < lips[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses bullish alignment or loses 1d uptrend
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses bearish alignment or loses 1d downtrend
            if not (jaw[i] < teeth[i] and teeth[i] < lips[i]) or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals