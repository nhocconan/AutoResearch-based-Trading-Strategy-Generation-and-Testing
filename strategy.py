#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian channel breakout with volume confirmation and daily trend filter
# Donchian(20) provides clear breakout levels, volume surge confirms momentum, daily EMA50 filters trend direction
# Works in bull markets via breakouts, in bear via mean reversion at channel boundaries
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_VolumeTrendFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA for trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily 50-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Calculate Donchian channel (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma_30)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper channel with volume confirmation and uptrend
            if close[i] > upper_channel[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower channel with volume confirmation and downtrend
            elif close[i] < lower_channel[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower channel (failed breakout) or reaches upper channel (take profit at channel top)
            if close[i] < lower_channel[i] or close[i] >= upper_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper channel (failed breakdown) or reaches lower channel (take profit at channel bottom)
            if close[i] > upper_channel[i] or close[i] <= lower_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals