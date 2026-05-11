#!/usr/bin/env python3
name = "6h_ElderRay_26EMA_Trend_VolumeSpike"
timeframe = "6h"
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
    
    # 1d EMA26 for trend (used in Elder Ray)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    ema_26_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_26_1d)
    
    # Elder Ray components: Bull Power = High - EMA26, Bear Power = Low - EMA26
    bull_power = high - ema_26_1d_aligned
    bear_power = low - ema_26_1d_aligned
    
    # 1d volume filter: volume > 1.8x 20-day average (to avoid false signals)
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    volume_filter = volume > 1.8 * vol_ma20_1d_aligned
    
    # 13-period EMA for entry/exit timing (on 6h close)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(26, 20, 13)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_26_1d_aligned[i]) or np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(ema_13[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bullish momentum) + price > EMA13 + volume spike
            if bull_power[i] > 0 and close[i] > ema_13[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bearish momentum) + price < EMA13 + volume spike
            elif bear_power[i] < 0 and close[i] < ema_13[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bull Power <= 0 or price < EMA13
            if bull_power[i] <= 0 or close[i] < ema_13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bear Power >= 0 or price > EMA13
            if bear_power[i] >= 0 or close[i] > ema_13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals