#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > 1d EMA50) and 1d volume spike confirmation.
# Long when price breaks above 20-bar Donchian high AND price > 1d EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below 20-bar Donchian low AND price < 1d EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Exit when price crosses 10-bar Donchian midpoint (mean reversion to reduce whipsaw).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing strong trending moves with volume confirmation in both bull and bear markets.
# Target: 60-100 total trades over 4 years (15-25/year) for 6h timeframe.

name = "6h_Donchian20_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume spike filter (HTF)
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    # Calculate Donchian channels (20-period) on primary timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > 1d EMA50 AND volume spike
            if (close[i] > highest_high[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):  # True if volume spike aligned
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND price < 1d EMA50 AND volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals