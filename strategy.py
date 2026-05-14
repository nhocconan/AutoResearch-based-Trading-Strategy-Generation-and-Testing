#!/usr/bin/env python3
# Hypothesis: 12h TRIX momentum with 1d EMA34 trend filter and 1w volume spike confirmation.
# Long when TRIX crosses above zero AND 1d close > EMA34 (uptrend) AND 1w volume > 2.0 * 20-period average volume.
# Short when TRIX crosses below zero AND 1d close < EMA34 (downtrend) AND 1w volume > 2.0 * 20-period average volume.
# Exit when TRIX reverses sign (crosses zero in opposite direction).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for 12h timeframe with strict entry conditions.
# TRIX is effective in both bull and bear markets as it filters noise and captures momentum shifts.
# Volume spike threshold increased to 2.0x to reduce trades and avoid overtrading.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_TRIX_ZeroCross_1dEMA34_1wVolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1w volume confirmation filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    volume_1w = df_1w['volume'].values
    vol_ma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1w = volume_1w > (2.0 * vol_ma_20_1w)
    volume_confirm_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_confirm_1w.astype(float))
    
    # Calculate TRIX (15-period EMA of EMA of EMA of close, then ROC)
    # TRIX = 100 * (EMA3(close) - EMA3(close)_prev) / EMA3(close)_prev
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # First value has no previous
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_confirm_1w_aligned[i]) or
            np.isnan(trix[i]) or
            np.isnan(trix[i-1])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = prices.index[i].hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero AND 1d close > 1d EMA34 (uptrend) AND volume confirmation
            if (trix[i-1] <= 0 and trix[i] > 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero AND 1d close < 1d EMA34 (downtrend) AND volume confirmation
            elif (trix[i-1] >= 0 and trix[i] < 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero (momentum reversal)
            if trix[i-1] > 0 and trix[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero (momentum reversal)
            if trix[i-1] < 0 and trix[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals