#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA trend filter and volume spike
# Uses daily trend alignment to avoid counter-trend trades, volume spike for confirmation
# Target: 15-25 trades/year per symbol (60-100 total) to stay within optimal range
# Donchian channels provide clear breakout levels, EMA filter ensures trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 12h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Highest high of last 20 periods
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lowest low of last 20 periods
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_1d[i]) or np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-period high + volume spike + uptrend (price > 1d EMA50)
            if (close[i] > high_max[i] and vol_spike[i] and close[i] > ema_50_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low + volume spike + downtrend (price < 1d EMA50)
            elif (close[i] < low_min[i] and vol_spike[i] and close[i] < ema_50_1d[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_min[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_max[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Volume_Spike"
timeframe = "12h"
leverage = 1.0