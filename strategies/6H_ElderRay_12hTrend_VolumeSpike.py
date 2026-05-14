#!/usr/bin/env python3
name = "6H_ElderRay_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # Volume spike filter: current volume > 2.0 x 20-period average
    vol_ma_20 = np.zeros_like(volume)
    vol_ma_20[:] = np.nan
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Bull Power > 0 + 12h uptrend + volume spike
            if bull_power[i] > 0 and close[i] > ema20_12h_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 + 12h downtrend + volume spike
            elif bear_power[i] < 0 and close[i] < ema20_12h_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power < 0 (momentum shift) or trend reversal
            if bear_power[i] < 0 or close[i] < ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power > 0 (momentum shift) or trend reversal
            if bull_power[i] > 0 or close[i] > ema20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals