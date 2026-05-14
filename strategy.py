#!/usr/bin/env python3
# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA50 trend filter and 1d volume confirmation.
# Long when Bull Power > 0 AND 1d close > 1d EMA50 (uptrend) AND 1d volume > 1.5 * 20-period average volume.
# Short when Bear Power < 0 AND 1d close < 1d EMA50 (downtrend) AND 1d volume > 1.5 * 20-period average volume.
# Exit when Elder Power reverses sign (Bull Power <= 0 for long exit, Bear Power >= 0 for short exit).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 6h timeframe with strict entry conditions to avoid overtrading.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

name = "6h_ElderRay_TrendFilter_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume confirmation filter (HTF)
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = volume_1d > (1.5 * vol_ma_20_1d)  # Volume > 1.5x 20-period MA
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate Elder Ray Index components (LTF)
    # Bull Power = High - EMA13(Close)
    # Bear Power = Low - EMA13(Close)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bull Power > 0 AND 1d close > 1d EMA50 (uptrend) AND volume confirmation
            if (bull_power[i] > 0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Bear Power < 0 AND 1d close < 1d EMA50 (downtrend) AND volume confirmation
            elif (bear_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_confirm_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bull Power <= 0 (momentum weakening)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bear Power >= 0 (momentum weakening)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals