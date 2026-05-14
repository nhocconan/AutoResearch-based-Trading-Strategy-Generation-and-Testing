#!/usr/bin/env python3
# Hypothesis: 4h Williams %R extremes with 12h EMA34 trend filter and volume confirmation (>1.8x 20-period average).
# Williams %R(14) < -80 = oversold (long setup), > -20 = overbought (short setup).
# Enter long when %R crosses above -80 from below AND close > 12h EMA34 (bullish trend) AND volume > 1.8x MA20.
# Enter short when %R crosses below -20 from above AND close < 12h EMA34 (bearish trend) AND volume > 1.8x MA20.
# Exit long when %R crosses above -20 (overbought) OR close < 12h EMA34 (trend change).
# Exit short when %R crosses below -80 (oversold) OR close > 12h EMA34 (trend change).
# Uses 12h HTF for trend to reduce noise and overtrading. Volume confirmation reduces false signals.
# Target: 80-180 total trades over 4 years (20-45/year) to stay within fee drag limits for 4h timeframe.
# Williams %R is effective in ranging and trending markets, capturing reversals at extremes.

name = "4h_WilliamsR_Extremes_12hEMA34_VolumeConfirm_v1"
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
    
    # --- 4h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Williams %R crossover signals
    williams_r_prev = np.roll(williams_r, 1)
    williams_r_prev[0] = np.nan
    williams_r_cross_above_80 = (williams_r_prev <= -80) & (williams_r > -80)
    williams_r_cross_below_20 = (williams_r_prev >= -20) & (williams_r < -20)
    
    # 4h volume confirmation: > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
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
            np.isnan(williams_r[i]) or
            np.isnan(williams_r_cross_above_80[i]) or
            np.isnan(williams_r_cross_below_20[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (oversold bounce) AND close > 12h EMA34 (bullish trend) AND volume confirm
            if (williams_r_cross_above_80[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (overbought rejection) AND close < 12h EMA34 (bearish trend) AND volume confirm
            elif (williams_r_cross_below_20[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -20 (overbought) OR close < 12h EMA34 (trend change)
            if (williams_r_cross_below_20[i] or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -80 (oversold) OR close > 12h EMA34 (trend change)
            if (williams_r_cross_above_80[i] or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals