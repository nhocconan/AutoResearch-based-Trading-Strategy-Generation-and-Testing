#!/usr/bin/env python3
# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-bar high AND 4h EMA50 rising AND 1d volume > 1.5x 20-day average volume.
# Short when price breaks below 20-bar low AND 4h EMA50 falling AND 1d volume > 1.5x 20-day average volume.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours. Fixed position size 0.20 to minimize fee churn.
# Designed for 15-35 trades/year by requiring multi-timeframe confluence and volume confirmation.
# Works in bull markets via breakout momentum and in bear markets via breakdown strength.

name = "1h_Donchian20_4hTrend_1dVolSpike_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h close for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume on 1d
    vol_avg_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_threshold = vol_avg_20d * 1.5
    volume_spike = volume_1d > vol_spike_threshold
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float), additional_delay_bars=0)
    
    # Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-bar high AND 4h EMA50 rising AND 1d volume spike
            if close[i] > highest_20[i] and ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and volume_spike_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 20-bar low AND 4h EMA50 falling AND 1d volume spike
            elif close[i] < lowest_20[i] and ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and volume_spike_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-bar low OR 4h EMA50 starts falling
            if close[i] < lowest_20[i] or ema50_4h_aligned[i] < ema50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-bar high OR 4h EMA50 starts rising
            if close[i] > highest_20[i] or ema50_4h_aligned[i] > ema50_4h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals