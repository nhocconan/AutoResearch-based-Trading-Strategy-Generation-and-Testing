#!/usr/bin/env python3
name = "4h_MR_Reversal_TRIX_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE for trend filter and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # TRIX (15-period) for mean reversion signal
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix_raw[0] = 0  # First value is invalid due to roll
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Daily close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily ATR for chop filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.roll(df_1d['close'], 1))
    tr3 = np.abs(df_1d['low'] - np.roll(df_1d['close'], 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily range for chop calculation
    range_1d = df_1d['high'] - df_1d['low']
    sum_range_14 = pd.Series(range_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_range_14 / (atr_14_1d * 14)) / np.log10(2)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detection on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(trix[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion conditions
        # Long when TRIX is deeply oversold in choppy market
        long_cond = (trix[i] < -1.5) and (chop_aligned[i] > 61.8)
        # Short when TRIX is deeply overbought in choppy market
        short_cond = (trix[i] > 1.5) and (chop_aligned[i] > 61.8)
        
        # Volume confirmation
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        # Trend filter: only trade against the daily trend
        # In uptrend (price > EMA34), only take short signals (fade strength)
        # In downtrend (price < EMA34), only take long signals (fade weakness)
        if position == 0:
            if long_cond and vol_condition and close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            elif short_cond and vol_condition and close[i] > ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit when TRIX returns to neutral or trend resumes
            if trix[i] > -0.5 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit when TRIX returns to neutral or trend resumes
            if trix[i] < 0.5 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h TRIX mean reversion with daily trend filter and chop filter
# - TRIX (15,9) identifies overbought/oversold conditions
# - Chop filter (Choppiness Index > 61.8) ensures ranging market for mean reversion
# - Daily EMA34 trend filter: only fade the trend (short in uptrend, long in downtrend)
# - Volume confirmation (1.5x average) validates the reversal signal
# - Works in both bull and bear markets by fading the intermediate trend
# - Position size 0.25 limits risk and reduces trade frequency
# - Target: 20-50 trades/year to avoid fee drag
# - Mean reversion works well in choppy, range-bound markets (common in 2025+)