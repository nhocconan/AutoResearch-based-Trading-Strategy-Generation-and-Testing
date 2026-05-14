#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.5x 20-period average).
# Long when price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume > 1.5x MA20.
# Short when price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume > 1.5x MA20.
# Exit when price retests the broken level (R3 for long, S3 for short) OR closes below/above 1d EMA34.
# Uses 1d HTF for trend to reduce noise and overtrading. Volume confirmation (>1.5x) reduces false signals.
# Target: 100-200 total trades over 4 years (25-50/year) to stay within fee drag limits for 4h timeframe.
# Camarilla levels derived from prior 1d range provide institutional support/resistance.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels from prior 1d: R3, S3
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_range = high_1d - low_1d
    r3 = close_1d + (1.1 * camarilla_range / 2)
    s3 = close_1d - (1.1 * camarilla_range / 2)
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Align Camarilla levels to 4h (completed 1d bar only)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 AND bullish trend AND volume confirm
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 AND bearish trend AND volume confirm
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Retest R3 (mean reversion) OR trend breakdown
            if (close[i] <= r3_aligned[i] or 
                close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Retest S3 (mean reversion) OR trend breakdown
            if (close[i] >= s3_aligned[i] or 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals