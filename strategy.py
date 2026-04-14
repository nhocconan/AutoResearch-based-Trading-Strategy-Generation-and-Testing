#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA Trend + 1d RSI Mean Reversion with Volume Spike
# Uses Kaufman's Adaptive Moving Average (KAMA) on 12h for trend direction
# 1d RSI(14) for mean reversion entries (oversold/overbought)
# Volume spike confirmation to filter false signals
# Designed to work in both bull and bear markets by combining trend filter with mean reversion
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for KAMA trend
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate KAMA (10, 2, 30) on 12h close
    close_12h = df_12h['close'].values
    change = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if False else np.abs(np.diff(close_12h, prepend=close_12h[0]))
    # Correct ER calculation: Efficiency Ratio = |net change| / sum of absolute changes
    abs_changes = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    net_change = np.abs(close_12h - np.roll(close_12h, 1))
    net_change[0] = 0
    er = np.zeros_like(close_12h, dtype=np.float64)
    for i in range(len(close_12h)):
        if i < 10:
            er[i] = np.nan
        else:
            sum_abs_changes = np.sum(abs_changes[i-9:i+1])  # 10-period sum
            if sum_abs_changes > 0:
                er[i] = net_change[i] / sum_abs_changes
            else:
                er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.full_like(close_12h, np.nan)
    kama[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_12h = kama
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # Load 1d data ONCE before loop for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI (14) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    # Wilder's smoothing
    for i in range(len(close_1d)):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike (20-period) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for KAMA, RSI, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price relative to 12h KAMA
        above_kama = price > kama_12h_aligned[i]
        
        if position == 0:
            # Long: RSI oversold (<30) with volume spike and uptrend
            if rsi_1d_aligned[i] < 30 and vol_ratio[i] > 1.5 and above_kama:
                position = 1
                signals[i] = position_size
            # Short: RSI overbought (>70) with volume spike and downtrend
            elif rsi_1d_aligned[i] > 70 and vol_ratio[i] > 1.5 and not above_kama:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI overbought (>70) or trend changes
            if rsi_1d_aligned[i] > 70 or price < kama_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI oversold (<30) or trend changes
            if rsi_1d_aligned[i] < 30 or price > kama_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_Trend_1dRSI_Volume_Spike"
timeframe = "12h"
leverage = 1.0