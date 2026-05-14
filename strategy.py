#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA34 trend filter and 6h volume confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND price > 12h EMA34 (bullish trend) AND 6h volume > 2.0x 20-period average.
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 12h EMA34 (bearish trend) AND 6h volume > 2.0x 20-period average.
# Exit on opposite Elder Ray condition (Bull Power < 0 for longs, Bear Power > 0 for shorts).
# Uses 12h HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (2.0x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.
# Elder Ray captures both trend and momentum, effective in both bull and bear markets when combined with HTF trend filter.

name = "6h_ElderRay_BullBearPower_12hEMA34_6hVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    # Bear Power = EMA13 - Low
    bear_power = ema_13 - low
    # 6h volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) - trend filter (smooth for 6h trading)
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND price > 12h EMA34 (bullish) AND 6h volume confirm
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bull Power < 0 AND Bear Power > 0 AND price < 12h EMA34 (bearish) AND 6h volume confirm
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power < 0 (momentum weakening)
            if bull_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power > 0 (momentum weakening)
            if bear_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals