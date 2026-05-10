#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Adaptive
# Hypothesis: 12-hour breakouts from daily Camarilla R1/S1 levels with daily trend filter (EMA34) and volume confirmation.
# Adaptive position sizing based on volatility regime (ATR ratio) to reduce risk in high volatility and increase in low volatility.
# Designed for 12h to achieve 12-37 trades/year, suitable for both bull and bear markets.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Adaptive"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily ATR for volatility regime filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ratio = atr_14 / pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ratio > 3.0, 3.0, atr_ratio)  # Cap at 3.0
    
    # Camarilla levels (based on previous day)
    def calculate_camarilla(h, l, c):
        typical = (h + l + c) / 3.0
        range_ = h - l
        R1 = c + (range_ * 1.1000 / 12)
        S1 = c - (range_ * 1.1000 / 12)
        return R1, S1
    
    R1 = np.full_like(close_1d, np.nan)
    S1 = np.full_like(close_1d, np.nan)
    for i in range(1, len(close_1d)):
        R1[i], S1[i] = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(atr_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Adaptive position size based on volatility regime
        # Low volatility (atr_ratio < 0.8): full size
        # High volatility (atr_ratio > 1.2): reduced size
        vol_factor = 1.0
        if atr_ratio_aligned[i] > 1.2:
            vol_factor = 0.5
        elif atr_ratio_aligned[i] < 0.8:
            vol_factor = 1.0
        else:
            vol_factor = 0.75  # Medium volatility
        
        base_size = 0.25
        adaptive_size = base_size * vol_factor
        
        if position == 0:
            # Long: price breaks above R1, above daily EMA34, strong volume
            if close[i] > R1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = adaptive_size
                position = 1
            # Short: price breaks below S1, below daily EMA34, strong volume
            elif close[i] < S1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -adaptive_size
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 or below daily EMA34
            if close[i] < S1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = adaptive_size
        elif position == -1:
            # Short exit: price rises above R1 or above daily EMA34
            if close[i] > R1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -adaptive_size
    
    return signals