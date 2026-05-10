# 12h_KAMA_Trend_Breakout_1dVOL
# Hypothesis: 12-hour breakouts using Kaufman Adaptive Moving Average (KAMA) as trend filter and direction signal, combined with daily volume confirmation and ATR-based volatility filter. 
# KAMA adapts to market noise, reducing false signals in ranging markets while capturing trends. Daily volume ensures breakout strength, and volatility filter avoids low-momentum periods.
# Designed for 12h timeframe to achieve 12-37 trades/year, suitable for both bull and bear markets by following adaptive trend and avoiding chop.

name = "12h_KAMA_Trend_Breakout_1dVOL"
timeframe = "12h"
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
    
    # Daily data for volume confirmation and ATR filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily ATR(14) for volatility filter
    def calculate_atr(high, low, close, period):
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = np.nan  # First TR is invalid
        atr = np.full_like(close, np.nan, dtype=np.float64)
        if len(close) >= period:
            # Initial ATR as simple average
            atr[period-1] = np.nanmean(tr[1:period])
            # Wilder smoothing
            for i in range(period, len(close)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # 12h KAMA(10,2,30) for trend and direction
    def kama(close, fast=2, slow=30, er_period=10):
        change = np.abs(np.diff(close, n=er_period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if er_period == 1 else \
                     np.array([np.sum(np.abs(np.diff(close[i-er_period+1:i+1]))) 
                               for i in range(er_period-1, len(close))])
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(slow+1) - 2/(fast+1)) + 2/(fast+1)) ** 2
        kama_vals = np.full_like(close, np.nan)
        kama_vals[er_period] = close[er_period]  # Seed
        for i in range(er_period+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i-er_period] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_12h = kama(close, fast=2, slow=30, er_period=10)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 10)  # Ensure enough history for KAMA and ATR
    
    for i in range(start_idx, n):
        if np.isnan(kama_12h[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility regimes (ATR < 50% of MA)
        if i >= 20:
            atr_ma = np.nanmean(atr_14_1d_aligned[i-19:i+1])  # 20-period ATR average
            if atr_14_1d_aligned[i] < 0.5 * atr_ma:
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
        
        if position == 0:
            # Long: price above KAMA and rising, with strong volume
            if close[i] > kama_12h[i] and kama_12h[i] > kama_12h[i-1] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and falling, with strong volume
            elif close[i] < kama_12h[i] and kama_12h[i] < kama_12h[i-1] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA or volatility drops
            if close[i] < kama_12h[i] or (i >= 20 and atr_14_1d_aligned[i] < 0.5 * atr_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA or volatility drops
            if close[i] > kama_12h[i] or (i >= 20 and atr_14_1d_aligned[i] < 0.5 * atr_ma):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals