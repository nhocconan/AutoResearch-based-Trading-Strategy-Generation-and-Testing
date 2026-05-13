#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h EMA34 trend filter + volume confirmation (>1.5x 20-bar avg). 
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 AND price > 12h EMA34 AND volume spike; Short when Bear Power < 0 AND price < 12h EMA34 AND volume spike.
# Exit when power reverses sign or volume drops. Designed to capture momentum shifts in both bull and bear markets.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee churn.

name = "6h_ElderRay_BullBearPower_12hEMA34_VolumeConfirm_v1"
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
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate EMA13 for Elder Ray (on 6h data)
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bulls in control), price > 12h EMA34 (uptrend), volume spike (>1.5x avg)
            if (bull_power[i] > 0 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bears in control), price < 12h EMA34 (downtrend), volume spike (>1.5x avg)
            elif (bear_power[i] < 0 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power turns negative (bulls lose control) OR volume drops below average
            if (bull_power[i] <= 0 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power turns positive (bears lose control) OR volume drops below average
            if (bear_power[i] >= 0 or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals