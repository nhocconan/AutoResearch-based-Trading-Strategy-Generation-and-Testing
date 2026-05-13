#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-period 12h Donchian high AND price > 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Short when price breaks below 20-period 12h Donchian low AND price < 1d EMA50 AND 1d volume > 1.5 * 20-period average volume.
# Exit when price crosses below 10-period 12h Donchian mid (for longs) or above 10-period 12h Donchian mid (for shorts).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing medium-term trends with volatility-based breakouts and volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Donchian20_Breakout_1dEMA50_1dVolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate 12h Donchian channels (20-period high/low, 10-period mid)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid_10 = (pd.Series(high_12h).rolling(window=10, min_periods=10).max().values + 
                       pd.Series(low_12h).rolling(window=10, min_periods=10).min().values) / 2
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike filter
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (1.5 * vol_ma_20)
    
    # Align HTF indicators to LTF
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    donchian_mid_10_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid_10)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(donchian_mid_10_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period Donchian high AND price > 1d EMA50 AND volume spike
            if (close[i] > donchian_high_20_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period Donchian low AND price < 1d EMA50 AND volume spike
            elif (close[i] < donchian_low_20_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 10-period Donchian mid
            if close[i] < donchian_mid_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 10-period Donchian mid
            if close[i] > donchian_mid_10_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals