# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot breakouts (R1/S1) with 1-day EMA(50) trend filter and volume > 1.5x average.
# Works in bull markets via breakout continuation and in bear via mean-reversion off extremes.
# Target: 50-150 total trades over 4 years (~12-37/year) on 12h timeframe.
# Uses discrete position sizing (±0.25) to limit turnover and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d EMA to 12h timeframe
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h pivot points (H, L, C from previous day)
    # We need previous day's high, low, close for Camarilla calculation
    # Since we're on 12h timeframe, we'll use daily data for pivot calculation
    pivot_high = df_1d['high'].values
    pivot_low = df_1d['low'].values
    pivot_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = Close + (High - Low) * 1.12 / 12
    # S1 = Close - (High - Low) * 1.12 / 12
    camarilla_range = pivot_high - pivot_low
    r1 = pivot_close + camarilla_range * 1.12 / 12
    s1 = pivot_close - camarilla_range * 1.12 / 12
    
    # Align pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need EMA (50), volume MA (20)
    start_idx = max(ema_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1d EMA(50)
        uptrend = price > ema_aligned[i]
        downtrend = price < ema_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above R1 in uptrend with volume
            if price > r1_aligned[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below S1 in downtrend with volume
            elif price < s1_aligned[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price falls below S1 or trend reverses
            if price < s1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price rises above R1 or trend reverses
            if price > r1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0