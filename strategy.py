#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout (20-period) with volume confirmation and 12h EMA trend filter
# Uses Donchian breakouts for momentum capture in both bull and bear markets
# Volume spike filters weak breakouts, 12h EMA ensures alignment with higher timeframe trend
# Target: 20-30 trades/year per symbol (80-120 total) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 20-period Donchian channels
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period EMA on 12h close for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_20_12h = close_12h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align 12h EMA to 4-hour timeframe
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume spike + uptrend (price > 12h EMA20)
            if (close[i] > high_max20[i] and vol_spike[i] and close[i] > ema_20_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume spike + downtrend (price < 12h EMA20)
            elif (close[i] < low_min20[i] and vol_spike[i] and close[i] < ema_20_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_min20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_max20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_12hEMA20_Trend"
timeframe = "4h"
leverage = 1.0