#!/usr/bin/env python3
# 1d_1w_kama_rsi_volume_filter_v1
# Hypothesis: Use 1-day KAMA trend with RSI momentum and volume confirmation on daily timeframe.
# KAMA adapts to market noise, reducing false signals in choppy markets. RSI filters overextended moves.
# Volume confirmation ensures institutional participation. Works in both bull and bear markets by
# following the adaptive trend while avoiding whipsaws. Target: 15-25 trades/year (60-100 total).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_volume_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = volumes = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) - adapts to market noise
    # ER = Efficiency Ratio = |net change| / sum of absolute changes
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if len(change) > 1 else np.array([0])
    # Calculate ER properly
    er = np.zeros_like(close)
    for i in range(1, len(close)):
        if i >= 1:
            net_change = np.abs(close[i] - close[i-9]) if i >= 9 else np.abs(close[i] - close[0])
            sum_changes = np.sum(np.abs(np.diff(close[max(0, i-9):i+1]))) if i >= 1 else 0
            er[i] = net_change / sum_changes if sum_changes > 0 else 0
    # Smooth ER with constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter from 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA(21) for trend
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema21_1w_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.3 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price below KAMA OR RSI overbought (>70) OR weekly trend turns down
            if close[i] < kama[i] or rsi[i] > 70 or close[i] < ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above KAMA OR RSI oversold (<30) OR weekly trend turns up
            if close[i] > kama[i] or rsi[i] < 30 or close[i] > ema21_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price above KAMA AND RSI not overbought (<60) AND volume surge AND weekly uptrend
            if (close[i] > kama[i] and rsi[i] < 60 and vol_surge and 
                close[i] > ema21_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price below KAMA AND RSI not oversold (>40) AND volume surge AND weekly downtrend
            elif (close[i] < kama[i] and rsi[i] > 40 and vol_surge and 
                  close[i] < ema21_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals