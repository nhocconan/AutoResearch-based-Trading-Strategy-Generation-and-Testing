#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 12h EMA34 trend filter and volume confirmation (>1.3x 20-bar avg volume). Elder Ray measures bull/bear strength relative to EMA13. Long when Bull Power > 0 and Bear Power < 0 (bulls in control) with uptrend filter. Short when Bear Power > 0 and Bull Power < 0 (bears in control) with downtrend filter. Uses 6h timeframe to target 50-150 total trades over 4 years. Discrete position sizing (0.25) minimizes fee churn. Works in bull/bear via trend filter.

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
    
    # Calculate EMA13 for Elder Ray (using LTF close)
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
        if (np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (strong bulls) AND Bear Power < 0 (weak bears) AND price > 12h EMA34 (uptrend) AND volume spike (>1.3x avg)
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 (strong bears) AND Bull Power < 0 (weak bulls) AND price < 12h EMA34 (downtrend) AND volume spike (>1.3x avg)
            elif (bear_power[i] > 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Either Bear Power becomes positive (bears take over) OR price crosses below 12h EMA34 (trend break)
            if (bear_power[i] > 0 or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Either Bull Power becomes positive (bulls take over) OR price crosses above 12h EMA34 (trend break)
            if (bull_power[i] > 0 or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals