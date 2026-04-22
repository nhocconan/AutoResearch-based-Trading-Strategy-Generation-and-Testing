#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h trend filter and volume confirmation
# Uses Donchian channels for breakout detection, filters by 12h EMA trend direction
# Requires volume spike for confirmation. Designed to work in both bull and bear markets
# by following higher timeframe trend. Target: 15-25 trades/year per symbol (60-100 total)
# to avoid fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    high_ma20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Load 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h close for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 6-hour timeframe
    high_ma20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), high_ma20)
    low_ma20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), low_ma20)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(high_ma20_aligned[i]) or np.isnan(low_ma20_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian + volume spike + uptrend (price > 12h EMA50)
            if (close[i] > high_ma20_aligned[i] and vol_spike[i] and close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + volume spike + downtrend (price < 12h EMA50)
            elif (close[i] < low_ma20_aligned[i] and vol_spike[i] and close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_ma20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_ma20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50_Volume_Spike_Session"
timeframe = "6h"
leverage = 1.0