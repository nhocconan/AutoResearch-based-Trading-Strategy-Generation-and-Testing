# 12H_KAMA_Trend_1D_TrendFilter_Volume
# Hypothesis: Use 12h KAMA trend direction filtered by 1d EMA trend and volume confirmation.
# KAMA adapts to market noise, reducing false signals in choppy markets.
# Long when: 12h KAMA rising, 1d EMA50 uptrend, volume > 1.3x average.
# Short when: 12h KAMA falling, 1d EMA50 downtrend, volume > 1.3x average.
# Works in bull/bear by following adaptive trend with volume confirmation.
# Target: 15-30 trades/year per symbol.

name = "12H_KAMA_Trend_1D_TrendFilter_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # 12h KAMA (adaptive moving average)
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[:10] = 0  # first 10 periods have no 10-period change
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder, will compute properly below
    # Recompute volatility correctly
    volatility = np.zeros_like(close)
    for i in range(1, len(close)):
        volatility[i] = volatility[i-1] + abs(close[i] - close[i-1])
    # For first 10 periods, use expanding window
    for i in range(10):
        volatility[i] = np.sum(np.abs(np.diff(close[:i+1])))
    er = np.where(volatility > 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA slope (1-period change)
    kama_slope = kama - np.roll(kama, 1)
    kama_slope[0] = 0
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 12h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_slope[i]) or np.isnan(vol_ma[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.3
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        
        if position == 0:
            # Enter long: daily uptrend + KAMA rising + volume
            if daily_up and kama_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: daily downtrend + KAMA falling + volume
            elif daily_down and kama_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when KAMA turns down or daily trend changes
            if not kama_up or not daily_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when KAMA turns up or daily trend changes
            if not kama_down or not daily_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals