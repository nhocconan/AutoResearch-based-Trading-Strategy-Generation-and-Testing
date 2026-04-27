# 4h CAMARILLA PIVOT R1 S1 BREAKOUT WITH VOLUME CONFIRMATION AND TREND FILTER
# Hypothesis: CAMARILLA PIVOT LEVELS ARE SIGNIFICANT SUPPORT/RESISTANCE LEVELS
# BREAKOUTS ABOVE R1 OR BELOW S1 WITH VOLUME CONFIRMATION AND TREND FILTER
# ARE HIGH PROBABILITY TRADES THAT WORK IN BOTH BULL AND BEAR MARKETS
# TREND FILTER: 1D EMA34 ENSURES WE TRADE IN DIRECTION OF HIGHER TIMEFRAME TREND
# VOLUME CONFIRMATION: VOLUME > 1.5X 20-PERIOD AVERAGE
# POSITION SIZE: 0.25 (25%) TO BALANCE RISK AND RETURN
# TARGET: 20-50 TRADES/YEAR TO AVOID FEE DRAG

#!/usr/bin/env python3
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
    
    # Get 1d data for CAMARILLA PIVOT AND EMA34 TREND FILTER
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CAMARILLA PIVOT LEVELS FROM PREVIOUS DAY
    # PIVOT = (HIGH + LOW + CLOSE) / 3
    # R1 = CLOSE + (HIGH - LOW) * 1.1 / 12
    # S1 = CLOSE - (HIGH - LOW) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Calculate 1-day EMA34 FOR TREND FILTER
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA FOR CONFIRMATION
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # ALIGN 1-DAY INDICATORS TO 4H TIMEFRAME
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: FLAT, 1: LONG, -1: SHORT
    size = 0.25   # 25% POSITION SIZE
    
    # WARMUP: NEED PIVOT, R1, S1, EMA34, AND VOLUME MA20
    start_idx = max(19, ema_period - 1)  # 20 for VOLUME, 33 FOR EMA
    
    for i in range(start_idx, n):
        # SKIP IF ANY DATA NOT READY
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # VOLUME FILTER
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # LONG: BREAK ABOVE R1 WITH UPTREND AND VOLUME
            if (price > r1_1d_aligned[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # SHORT: BREAK BELOW S1 WITH DOWNTREND AND VOLUME
            elif (price < s1_1d_aligned[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: PRICE CROSSES BELOW PIVOT OR STOPLOSS
            if price < pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # EXIT SHORT: PRICE CROSSES ABOVE PIVOT OR STOPLOSS
            if price > pivot_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0