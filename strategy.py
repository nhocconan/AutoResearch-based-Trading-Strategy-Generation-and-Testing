#!/usr/bin/env python3
# 4H_KAMA_REVERSAL_12H_TREND_VOLUME
# Hypothesis: KAMA on 4h detects mean-reversion opportunities when price deviates significantly from trend.
# Combines with 12h trend filter and volume confirmation to capture reversals in both bull and bear markets.
# Uses adaptive smoothing to reduce whipsaw in ranging markets while capturing strong reversals.
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years).

name = "4H_KAMA_REVERSAL_12H_TREND_VOLUME"
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
    volume = prices['volume'].values
    
    # 12h data for trend filter and volume average
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h
    # ER (Efficiency Ratio) = |net change| / sum(|changes|)
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.subtract(close, np.roll(close, 1)))
    volatility = np.convolve(change, np.ones(10), mode='full')[:len(change)]  # 10-period volatility
    volatility[0] = change[0]  # handle first element
    er = np.where(volatility != 0, direction / volatility, 0)
    # Smoothing constants: fastest = 2/(2+1) = 0.67, slowest = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # EMA34 on 12h for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 12h
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(kama_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average
        volume_confirm = volume[i] > 1.5 * vol_ma_12h_aligned[i]
        
        if position == 0:
            # LONG: Price significantly below KAMA (oversold) in uptrend
            if (close[i] < kama_aligned[i] * 0.98 and  # 2% below KAMA
                close[i] > ema34_12h_aligned[i] and    # Above 12h EMA (uptrend)
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # SHORT: Price significantly above KAMA (overbought) in downtrend
            elif (close[i] > kama_aligned[i] * 1.02 and  # 2% above KAMA
                  close[i] < ema34_12h_aligned[i] and    # Below 12h EMA (downtrend)
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to KAMA or trend reversal
            if (close[i] >= kama_aligned[i] * 0.995 or  # Near KAMA
                close[i] <= ema34_12h_aligned[i]):      # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to KAMA or trend reversal
            if (close[i] <= kama_aligned[i] * 1.005 or  # Near KAMA
                close[i] >= ema34_12h_aligned[i]):      # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals