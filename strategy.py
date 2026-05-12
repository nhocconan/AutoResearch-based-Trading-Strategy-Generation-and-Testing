#!/usr/bin/env python3
# 6h_Keltner_Channel_Breakout_12hTrend_Volume
# Hypothesis: Keltner Channel breakouts on 6h with 12h EMA trend filter and volume confirmation.
# Uses ATR-based channel for adaptive volatility filtering. Works in bull (breakouts continue) 
# and bear (mean-reversion at extremes via trend filter). Target: 15-35 trades/year.

name = "6h_Keltner_Channel_Breakout_12hTrend_Volume"
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
    
    # === 12h Trend Filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 20-period EMA on 12h for trend direction
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === Keltner Channel (20, 2.0) on 6h ===
    # ATR(20)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # 20-period EMA as middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_keltner = ema_20 + 2.0 * atr_20
    lower_keltner = ema_20 - 2.0 * atr_20
    
    # === Volume Confirmation (20-period average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend direction
        trend_up = close[i] > ema_20_12h_aligned[i]
        trend_down = close[i] < ema_20_12h_aligned[i]
        
        # Volume filter: above average
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above upper Keltner with volume and uptrend
            if (close[i] > upper_keltner[i] and vol_ok and trend_up):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Keltner with volume and downtrend
            elif (close[i] < lower_keltner[i] and vol_ok and trend_down):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price returns to middle line or trend changes
            if (close[i] < ema_20[i] or not trend_up):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to middle line or trend changes
            if (close[i] > ema_20[i] or not trend_down):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals