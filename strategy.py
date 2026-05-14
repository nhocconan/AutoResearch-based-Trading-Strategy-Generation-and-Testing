#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA34 trend filter and 6h volume confirmation (>1.5x 20-period average).
# Bull Power = High - EMA13; Bear Power = EMA13 - Low.
# Long when Bull Power > 0 AND close > 12h EMA34 (bullish trend) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND close < 12h EMA34 (bearish trend) AND volume > 1.5x 20-period average.
# Exit when Bull Power <= 0 (for long) or Bear Power <= 0 (for short) OR price crosses 12h EMA34.
# Uses 12h HTF for trend to reduce noise and overtrading vs shorter trends. Volume confirmation (1.5x) reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.
# Elder Ray measures bull/bear power behind price moves, effective in both bull and bear markets when combined with HTF trend filter.

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
    # 6h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.5 * vol_ma_20)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = ema_13 - low
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(34) - trend filter
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
            # LONG: Bull Power > 0 AND close > 12h EMA34 (bullish trend) AND volume confirm
            if (bull_power[i] > 0 and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power > 0 AND close < 12h EMA34 (bearish trend) AND volume confirm
            elif (bear_power[i] > 0 and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (weakening bulls) OR price crosses below 12h EMA34 (trend change)
            if (bull_power[i] <= 0 or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power <= 0 (weakening bears) OR price crosses above 12h EMA34 (trend change)
            if (bear_power[i] <= 0 or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals