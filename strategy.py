#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 12h EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 and Bear Power < 0 (bullish momentum), price > 12h EMA50, volume spike. Short when Bear Power < 0 and Bull Power < 0 (bearish momentum), price < 12h EMA50, volume spike. Exit when momentum diverges (Bull Power < 0 for long, Bear Power > 0 for short) or opposite volume spike. Designed for BTC/ETH robustness: Elder Ray captures power of bulls/bears, EMA50 filter ensures trend alignment, volume confirmation avoids false breakouts. Targets 12-37 trades/year on 6h timeframe.

name = "6h_ElderRay_TrendFilter_VolumeConfirm_v1"
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
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bulls in control), Bear Power < 0 (bears weak), price > 12h EMA50, volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bears in control), Bull Power < 0 (bulls weak), price < 12h EMA50, volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (bulls losing control) OR Bear Power >= 0 (bears taking over)
            if (bull_power[i] <= 0 or 
                bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (bears losing control) OR Bull Power <= 0 (bulls taking over)
            if (bear_power[i] >= 0 or 
                bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals