#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA34 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 and rising, Bear Power < 0 and falling, price > 1d EMA34, volume > 1.3x avg.
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, price < 1d EMA34, volume > 1.3x avg.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Elder Ray measures bull/bear strength relative to EMA13. Combined with 1d EMA34 trend filter, it avoids counter-trend trades.
# Volume confirmation ensures institutional participation. Works in bull markets via upward Elder Ray strength and in bear markets via downward strength.

name = "6h_ElderRay_EMA13_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe (wait for 1d bar to close)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 and rising (bullish momentum), price > 1d EMA34 (uptrend), volume > 1.3x average
            if (bull_power[i] > 0 and 
                i > 20 and bull_power[i] > bull_power[i-1] and  # Rising bull power
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 1.3 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 and rising (bearish momentum), price < 1d EMA34 (downtrend), volume > 1.3x average
            elif (bear_power[i] > 0 and 
                  i > 20 and bear_power[i] > bear_power[i-1] and  # Rising bear power
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 1.3 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power > 0 and rising (bearish momentum taking over) OR price < 1d EMA34
            if (bear_power[i] > 0 and 
                i > 20 and bear_power[i] > bear_power[i-1]) or \
               close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power > 0 and rising (bullish momentum taking over) OR price > 1d EMA34
            if (bull_power[i] > 0 and 
                i > 20 and bull_power[i] > bull_power[i-1]) or \
               close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals