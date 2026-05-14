#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above Donchian upper band AND 1w EMA50 is bullish (price > EMA50) AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below Donchian lower band AND 1w EMA50 is bearish (price < EMA50) AND 1d volume > 2.0 * 20-period average volume.
# Exit when price retraces to the midline of the Donchian channel (average of upper and lower bands).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing medium-term breakouts with volume confirmation in trending markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Donchian20_Breakout_1wEMA50_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian(20) channels (using primary timeframe data)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC (using prices.index which is DatetimeIndex)
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian upper band AND 1w EMA50 bullish (price > EMA50) AND volume spike
            if (open_[i] <= highest_20[i] and close[i] > highest_20[i] and 
                close[i] > ema50_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian lower band AND 1w EMA50 bearish (price < EMA50) AND volume spike
            elif (open_[i] >= lowest_20[i] and close[i] < lowest_20[i] and 
                  close[i] < ema50_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price retraces to Donchian midline
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price retraces to Donchian midline
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals