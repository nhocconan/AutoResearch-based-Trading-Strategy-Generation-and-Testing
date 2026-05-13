#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation (>1.5x 20-bar avg). Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 and Bear Power < 0 (bullish momentum) with price > 1d EMA50 and volume spike. Short when Bear Power > 0 and Bull Power < 0 (bearish momentum) with price < 1d EMA50 and volume spike. Exits when momentum diverges or volume drops. Designed for BTC/ETH robustness: Elder Ray captures institutional buying/selling pressure, EMA50 filter ensures trend alignment, volume confirmation avoids false signals. Targets 12-37 trades/year on 6h timeframe.

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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 and Bear Power < 0 (bullish momentum), price > 1d EMA50, volume spike (>1.5x avg)
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 and Bull Power < 0 (bearish momentum), price < 1d EMA50, volume spike (>1.5x avg)
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Momentum divergence (Bear Power > 0) OR volume drop (< avg_volume)
            if (bear_power[i] > 0 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Momentum divergence (Bull Power > 0) OR volume drop (< avg_volume)
            if (bull_power[i] > 0 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals