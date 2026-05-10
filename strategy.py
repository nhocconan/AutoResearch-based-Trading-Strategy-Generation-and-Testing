#!/usr/bin/env python3
# 1d_KAMA_Direction_WeeklyTrend_Volume
# Hypothesis: KAMA trend on daily, confirmed by weekly EMA trend and volume spike.
# Long when KAMA rising and weekly EMA trending up, short when opposite.
# Volume must be >1.5x average for entry. Exits when KAMA direction reverses.
# Designed for 10-25 trades/year to avoid fee drag, works in bull/bear via weekly trend filter.

name = "1d_KAMA_Direction_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Calculate KAMA (ER=10, SC=2,30) - Kaufman Adaptive Moving Average
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(change.shape) > 0 else np.abs(np.diff(close)).sum()
        # For simplicity in 1D array:
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.zeros_like(close)
        for i in range(1, len(close)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        
        er = np.zeros(n)
        for i in range(er_length, n):
            if volatility[i] != 0:
                er[i] = np.abs(close[i] - close[i-er_length]) / volatility[i]
            else:
                er[i] = 0
        
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_vals = np.zeros(n)
        kama_vals[0] = close[0]
        for i in range(1, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_vals = kama(close, 10, 2, 30)
    kama_dir = np.zeros(n)  # 1 for rising, -1 for falling
    for i in range(1, n):
        if kama_vals[i] > kama_vals[i-1]:
            kama_dir[i] = 1
        elif kama_vals[i] < kama_vals[i-1]:
            kama_dir[i] = -1
        else:
            kama_dir[i] = kama_dir[i-1]
    
    # Get weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama_vals[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Trade only in direction of weekly EMA50 trend
            if close[i] > ema_50_1w_aligned[i]:  # Uptrend
                # Long: KAMA rising with volume confirmation
                if kama_dir[i] == 1 and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Downtrend
                # Short: KAMA falling with volume confirmation
                if kama_dir[i] == -1 and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: KAMA turns down
            if kama_dir[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up
            if kama_dir[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals