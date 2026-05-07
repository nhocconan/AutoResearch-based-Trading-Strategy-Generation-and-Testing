#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w EMA20 filter and volume spike for entry.
# Long when KAMA turns upward and price > 1w EMA20 with volume spike.
# Short when KAMA turns downward and price < 1w EMA20 with volume spike.
# Uses 1w EMA20 trend filter to avoid counter-trend trades and stay aligned with higher timeframe.
# Volume spike ensures momentum confirmation. Designed for low trade frequency (target: 15-25/year) to minimize fee drag.
# KAMA adapts to market noise, making it effective in both trending and ranging markets.
# Works in both bull and bear markets by following the 1w trend direction.
name = "1d_KAMA_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w trend filter: 20-period EMA on close
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # KAMA calculation on 1d
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Avoid division by zero
    er = np.where(vol > 0, change / vol, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 1d volume average for spike detection
    vol_ema_1d = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = np.where(vol_ema_1d > 0, volume / vol_ema_1d, 1.0) > 1.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for KAMA and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(kama[i-1]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA direction: upward if current > previous, downward if current < previous
        kama_up = kama[i] > kama[i-1]
        kama_down = kama[i] < kama[i-1]
        
        # Trend filter: price above/below 1w EMA20
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long condition: KAMA turning up, in uptrend with volume spike
            long_condition = kama_up and uptrend and vol_spike[i]
            # Short condition: KAMA turning down, in downtrend with volume spike
            short_condition = kama_down and downtrend and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA turns down or trend turns down
            if (not kama_up) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA turns up or trend turns up
            if (not kama_down) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals