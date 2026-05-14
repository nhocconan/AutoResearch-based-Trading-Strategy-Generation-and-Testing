#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA34 trend filter and volume confirmation (>1.5x 20-period average).
# Bull Power = High - EMA13, Bear Power = Low - EMA13. Long when Bull Power > 0 AND close > 12h EMA34 (uptrend) AND volume confirm.
# Short when Bear Power < 0 AND close < 12h EMA34 (downtrend) AND volume confirm.
# Exit when power crosses zero or price crosses 12h EMA34 opposite.
# Uses 12h HTF for trend to reduce noise and overtrading. Volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
# Elder Ray measures bull/bear strength relative to EMA13; combined with 12h trend and volume gives high-conviction entries.

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
    # EMA13 for Elder Ray power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # High minus EMA13
    bear_power = low - ema13   # Low minus EMA13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) - trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema34_12h_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 (bullish strength) AND close > 12h EMA34 (uptrend) AND volume confirm
            if (bull_power[i] > 0 and 
                close[i] > ema34_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 (bearish strength) AND close < 12h EMA34 (downtrend) AND volume confirm
            elif (bear_power[i] < 0 and 
                  close[i] < ema34_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (loss of bullish strength) OR close < 12h EMA34 (trend change)
            if (bull_power[i] <= 0 or 
                close[i] < ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (loss of bearish strength) OR close > 12h EMA34 (trend change)
            if (bear_power[i] >= 0 or 
                close[i] > ema34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals