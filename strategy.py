#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA trend filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper band AND 12h EMA50 is rising AND 1d volume > 1.8 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 12h EMA50 is falling AND 1d volume > 1.8 * 20-period average volume.
# Exit when price retraces to the Donchian midpoint (mean of upper and lower bands).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing breakouts with volume confirmation in trending markets.
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Donchian20_Breakout_12hEMA50_1dVolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on primary timeframe
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    # Calculate EMA slope (rising/falling)
    ema_slope = np.diff(ema_50_aligned, prepend=ema_50_aligned[0])
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.8 * vol_ma_20)  # Volume spike condition
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(ema_slope[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND EMA50 rising AND volume spike
            if (close[i-1] <= highest_high[i-1] and close[i] > highest_high[i] and 
                ema_slope[i] > 0 and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND EMA50 falling AND volume spike
            elif (close[i-1] >= lowest_low[i-1] and close[i] < lowest_low[i] and 
                  ema_slope[i] < 0 and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals