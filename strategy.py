#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w trend filter and volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) with weekly uptrend (price > weekly EMA34) and 6h volume > 1.5x 20-period average.
# Short when Bear Power < 0 (low < EMA13) AND Bull Power < 0 (close < EMA13) with weekly downtrend (price < weekly EMA34) and 6h volume > 1.5x 20-period average.
# Exit when Elder Power signals reverse (Bull Power <= 0 for longs, Bear Power >= 0 for shorts).
# Uses 6h timeframe for lower frequency, weekly trend filter for major regime, Elder Ray for intrinsic bull/bear power measurement.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_ElderRay_1wEMA34_VolumeConfirm"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = Close - EMA13
    bull_power = close - ema_13
    # Bear Power = Low - EMA13
    bear_power = low - ema_13
    # 6h volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (1.5 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(34) - trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 (strong bullish) + weekly uptrend + volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_1w_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND Bull Power < 0 (strong bearish) + weekly downtrend + volume spike
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (loss of bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (loss of bearish momentum)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals