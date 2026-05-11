#!/usr/bin/env python3
# 4h_Vortex_Trend_Signal_1dTrend_Confirmation
# Hypothesis: Uses Vortex Indicator (VI+) and (VI-) to determine trend direction on 4h,
# confirmed by 1d EMA50 trend. Only takes long when VI+ > VI- and price > 1d EMA50,
# short when VI- > VI+ and price < 1d EMA50. Includes volatility filter using ATR to
# avoid choppy markets. Designed for low turnover in both bull and bear markets by
# requiring strong trend alignment across timeframes.

name = "4h_Vortex_Trend_Signal_1dTrend_Confirmation"
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
    
    # === Vortex Indicator on 4h (14-period) ===
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])],
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # VM+: |current high - prior low|, VM-: |current low - prior high|
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.abs(high[0] - low[0])  # first period
    vm_minus[0] = np.abs(low[0] - high[0])
    
    # Smooth using Wilder's smoothing (EMA-like with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    vip = pd.Series(vm_plus).ewm(alpha=1/14, adjust=False).mean().values
    vim = pd.Series(vm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    vi_plus = vip / (atr + 1e-10)
    vi_minus = vim / (atr + 1e-10)
    
    # === 1d EMA50 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volatility Filter: Avoid extreme ATR (chop) ===
    atr_ma = pd.Series(atr).ewm(span=20, adjust=False).mean().values
    atr_ratio = atr / (atr_ma + 1e-10)
    volatility_ok = (atr_ratio > 0.5) & (atr_ratio < 2.0)  # Avoid too low or too high volatility
    
    # === Signal Logic ===
    position_size = 0.25
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensures Vortex and EMA50 are valid
    
    for i in range(start_idx, n):
        if np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or np.isnan(ema50_1d_4h[i]) or np.isnan(volatility_ok[i]):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish trend) AND price above 1d EMA50 AND volatility OK
            if (vi_plus[i] > vi_minus[i] and 
                close[i] > ema50_1d_4h[i] and 
                volatility_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: VI- > VI+ (bearish trend) AND price below 1d EMA50 AND volatility OK
            elif (vi_minus[i] > vi_plus[i] and 
                  close[i] < ema50_1d_4h[i] and 
                  volatility_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit when trend weakens: VI+ and VI- cross in opposite direction
            if position == 1:
                if vi_minus[i] > vi_plus[i]:  # trend turns bearish
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if vi_plus[i] > vi_minus[i]:  # trend turns bullish
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals