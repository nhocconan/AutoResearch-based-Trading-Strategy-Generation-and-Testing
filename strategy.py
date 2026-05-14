#!/usr/bin/env python3
# Hypothesis: 12h Williams %R extreme levels with 1w EMA(50) trend filter and 12h volume confirmation.
# Long when Williams %R crosses above -80 (oversold recovery) with 1w EMA(50) bullish and volume > 1.5x 20-period average.
# Short when Williams %R crosses below -20 (overbought rejection) with 1w EMA(50) bearish and volume > 1.5x 20-period average.
# Exit on Williams %R crossing opposite extreme (-20 for longs, -80 for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn. Williams %R provides mean-reversion edge in ranging markets,
# while 1w EMA ensures trend alignment to avoid counter-trend trades. Volume confirmation filters weak breakouts.
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe.
# Works in bull/bear: 1w EMA filter avoids counter-trend trades in strong trends, Williams %R captures reversals in ranges.

name = "12h_WilliamsR_Extreme_1wEMA50_12hVolumeConfirm"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Williams %R (14-period) ---
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # --- 12h volume confirmation: > 1.5x 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1w EMA(50) (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold recovery) + 1w EMA bullish + volume confirm
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and
                close[i] > ema_50_1w_aligned[i] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought rejection) + 1w EMA bearish + volume confirm
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and
                  close[i] < ema_50_1w_aligned[i] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses below -20 (overbought territory)
            if williams_r[i] < -20 and williams_r[i-1] >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses above -80 (oversold territory)
            if williams_r[i] > -80 and williams_r[i-1] <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals