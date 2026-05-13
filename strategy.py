#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 regime filter and volume spike confirmation.
# Elder Ray measures bull/bear power relative to EMA13. In bull regime (price > 1d EMA34), we go long when bull power > 0 and volume spikes.
# In bear regime (price < 1d EMA34), we go short when bear power < 0 and volume spikes.
# Uses 6h timeframe for entries, 1d for regime and volume average. Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by adapting to the 1d trend regime.

name = "6h_ElderRay_BullBearPower_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d average volume for confirmation (20-period)
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 20:
        return np.zeros(n)
    volume_1d = df_1d_vol['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d_vol, avg_volume_1d)
    
    # Calculate EMA13 for Elder Ray (on 6h primary TF)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_1d_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull regime (price > 1d EMA34), bull power > 0, volume spike (>1.5x 1d avg)
            if (close[i] > ema_34_1d_aligned[i] and 
                bull_power[i] > 0 and 
                volume[i] > 1.5 * avg_volume_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear regime (price < 1d EMA34), bear power < 0, volume spike (>1.5x 1d avg)
            elif (close[i] < ema_34_1d_aligned[i] and 
                  bear_power[i] < 0 and 
                  volume[i] > 1.5 * avg_volume_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if regime turns bearish or volume drops
            if (close[i] < ema_34_1d_aligned[i]) or (volume[i] < 0.5 * avg_volume_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close if regime turns bullish or volume drops
            if (close[i] > ema_34_1d_aligned[i]) or (volume[i] < 0.5 * avg_volume_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals