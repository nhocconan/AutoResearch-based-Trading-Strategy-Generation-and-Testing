#!/usr/bin/env python3
# 4h_1d_1w_TRIX_Momentum_Regime
# Hypothesis: TRIX (12-period triple-smoothed EMA) detects momentum shifts.
# Long when TRIX crosses above zero with bullish weekly trend (price > weekly EMA50) and volume confirmation.
# Short when TRIX crosses below zero with bearish weekly trend (price < weekly EMA50) and volume confirmation.
# Uses 1d volatility filter (ATR ratio) to avoid choppy markets.
# Designed for low trade frequency (<100/year) to minimize fee drag.
# Works in bull/bear markets by aligning with weekly trend while using TRIX for precise momentum entries.

name = "4h_1d_1w_TRIX_Momentum_Regime"
timeframe = "4h"
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
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # TRIX on 4h close (triple EMA of EMA of EMA, then ROC)
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = 100 * (ema3.pct_change(periods=1))
    trix_signal = trix.values
    trix_prev = np.roll(trix_signal, 1)
    trix_prev[0] = np.nan
    
    # Daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and ATR(14)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR
    atr_ma = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1d / atr_ma
    low_volatility = atr_ratio < 0.8  # Avoid high volatility/chop
    
    # Weekly trend filter: price > weekly EMA50 for uptrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Align TRIX and volatility filter to 4h
    trix_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), trix_signal)
    trix_prev_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), trix_prev)
    low_volatility_aligned = align_htf_to_ltf(prices, df_1d, low_volatility)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(trix_aligned[i]) or np.isnan(trix_prev_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(low_volatility_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: TRIX crosses above zero + weekly uptrend + low volatility + volume spike
            if (trix_prev_aligned[i] <= 0 and trix_aligned[i] > 0 and
                close[i] > ema_50_1w_aligned[i] and
                low_volatility_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: TRIX crosses below zero + weekly downtrend + low volatility + volume spike
            elif (trix_prev_aligned[i] >= 0 and trix_aligned[i] < 0 and
                  close[i] < ema_50_1w_aligned[i] and
                  low_volatility_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero OR price closes below weekly EMA50
            if (trix_aligned[i] < 0) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero OR price closes above weekly EMA50
            if (trix_aligned[i] > 0) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals