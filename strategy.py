#!/usr/bin/env python3
"""
4h_1D_AdaptiveKeltner_DoubleEMA_Trend
Hypothesis: Price must break beyond Keltner Channel (ATR-based) with confirmation from dual EMA trend filter.
Long: Close > Upper Keltner + EMA20 > EMA50 + daily EMA50 uptrend.
Short: Close < Lower Keltner + EMA20 < EMA50 + daily EMA50 downtrend.
Exit: Price re-enters Keltner Channel or EMA20/EMA50 crossover reverses.
Uses tight entry conditions to limit trades (target 25-40/year) and works in both bull/bear markets.
ATR-based bands adapt to volatility, reducing false signals in low-vol periods.
"""

name = "4h_1D_AdaptiveKeltner_DoubleEMA_Trend"
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
    
    # ATR for Keltner Channel (20-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # EMA20 and EMA50 for trend filter
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Keltner Channel: EMA20 ± 2.0 * ATR
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # Daily data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(keltner_upper[i]) or 
            np.isnan(keltner_lower[i]) or 
            np.isnan(ema20[i]) or 
            np.isnan(ema50[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close > Upper Keltner + EMA20 > EMA50 + daily EMA50 uptrend
            if (close[i] > keltner_upper[i] and 
                ema20[i] > ema50[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close < Lower Keltner + EMA20 < EMA50 + daily EMA50 downtrend
            elif (close[i] < keltner_lower[i] and 
                  ema20[i] < ema50[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close < Lower Keltner OR EMA20 < EMA50
            if close[i] < keltner_lower[i] or ema20[i] < ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close > Upper Keltner OR EMA20 > EMA50
            if close[i] > keltner_upper[i] or ema20[i] > ema50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals