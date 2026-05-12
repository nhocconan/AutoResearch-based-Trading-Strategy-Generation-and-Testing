# 4h_12h_1d_ElderRay_Power_Trend
# Hypothesis: Uses Elder Ray Bull/Bear Power on 1d with 12h trend filter and volume spike.
# Bull Power = High - EMA(13), Bear Power = Low - EMA(13).
# Enters long when Bull Power > 0 (bullish pressure) with 12h uptrend and volume spike.
# Enters short when Bear Power < 0 (bearish pressure) with 12h downtrend and volume spike.
# Uses 12h EMA50 as trend filter to avoid counter-trend trades.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 12h trend while using 1d Elder Ray for precise entries.

name = "4h_12h_1d_ElderRay_Power_Trend"
timeframe = "4h"
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
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(13) for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bullish pressure) + 12h EMA50 uptrend + volume spike
            if (bull_power_aligned[i] > 0 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish pressure) + 12h EMA50 downtrend + volume spike
            elif (bear_power_aligned[i] < 0 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bear Power < 0 OR closes below 12h EMA50
            if (bear_power_aligned[i] < 0) or \
               (close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bull Power > 0 OR closes above 12h EMA50
            if (bull_power_aligned[i] > 0) or \
               (close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals