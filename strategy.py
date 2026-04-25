#!/usr/bin/env python3
"""
12h_KAMA_Trend_1dRSI_Extreme_VolumeFilter
Hypothesis: On 12h timeframe, KAMA trend direction combined with 1d RSI extremes (<30 for long, >70 for short) and volume spike (>2.0x 20-bar avg) captures mean-reversion moves within the primary trend. This strategy works in both bull and bear markets by trading pullbacks in trending markets (long in uptrend pullbacks, short in downtrend bounces). Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
"""

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
    
    # Get 1d data for HTF RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on 12h for trend filter
    # Efficiency Ratio (ER) = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))).cumsum()  # inefficient but correct concept
    # Proper ER calculation:
    er = np.zeros(n)
    for i in range(10, n):  # ER period = 10
        if i >= 10:
            net_change = abs(close[i] - close[i-10])
            sum_abs_changes = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = net_change / sum_abs_changes if sum_abs_changes > 0 else 0
    # Smoothing constants: fastest SC = 2/(2+1) = 0.6667, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.full(n, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_trend = np.where(close > kama, 1, -1)  # 1: uptrend, -1: downtrend
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(14, 20)  # RSI14, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_trend[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        trend = kama_trend[i]
        rsi_val = rsi_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        
        # Volume spike condition: current volume > 2.0x 20-period average
        volume_spike = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: KAMA trend + RSI extreme + volume spike
            # Long: uptrend (close > KAMA) + RSI < 30 (oversold) + volume spike
            long_signal = (trend == 1) and (rsi_val < 30) and volume_spike
            # Short: downtrend (close < KAMA) + RSI > 70 (overbought) + volume spike
            short_signal = (trend == -1) and (rsi_val > 70) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. RSI returns to neutral (>50) or trend changes
            if rsi_val > 50 or trend == -1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. RSI returns to neutral (<50) or trend changes
            if rsi_val < 50 or trend == 1:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "12h_KAMA_Trend_1dRSI_Extreme_VolumeFilter"
timeframe = "12h"
leverage = 1.0