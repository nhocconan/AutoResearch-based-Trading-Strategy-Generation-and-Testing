#!/usr/bin/env python3
# Hypothesis: 6h Williams %R Extreme Reversal with 12h EMA50 trend filter and volume confirmation (>1.8x 20-period average).
# Williams %R measures overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
# Long when Williams %R crosses above -80 from below (oversold reversal) AND close > 12h EMA50 (bullish trend) AND volume > 1.8x MA20.
# Short when Williams %R crosses below -20 from above (overbought reversal) AND close < 12h EMA50 (bearish trend) AND volume > 1.8x MA20.
# Exit when Williams %R crosses above -50 (for long) or below -50 (for short) OR price crosses 12h EMA50 in opposite direction.
# Uses 12h HTF for trend alignment to reduce whipsaws. Volume confirmation filters low-momentum false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.
# Williams %R is effective in ranging markets and captures reversals at extremes, complementing trend filter.

name = "6h_WilliamsR_Extreme_12hEMA50_6hVolumeConfirm_v1"
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
    
    # --- 6h Indicators (LTF) ---
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Replace division by zero or invalid with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 6h volume confirmation: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm_6h = volume > (1.8 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) - trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(williams_r[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_confirm_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R crosses above -80 (from below) AND close > 12h EMA50 (bullish trend) AND volume confirm
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_confirm_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R crosses below -20 (from above) AND close < 12h EMA50 (bearish trend) AND volume confirm
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_confirm_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R crosses above -50 (momentum weakening) OR price < 12h EMA50 (trend change)
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R crosses below -50 (momentum weakening) OR price > 12h EMA50 (trend change)
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals