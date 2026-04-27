#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Camarilla pivot levels (S3/R3) breakout with volume confirmation and 1-week EMA50 trend filter.
# Long when price breaks above R3 with 1w EMA50 uptrend and volume > 2x average.
# Short when price breaks below S3 with 1w EMA50 downtrend and volume > 2x average.
# Exit when price crosses the central pivot point (CPP).
# Uses tight entry conditions to limit trades and avoid fee drag, targeting 12-37 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (S3, R3)
    camarilla_S3 = np.full(len(close_1d), np.nan)
    camarilla_R3 = np.full(len(close_1d), np.nan)
    camarilla_CPP = np.full(len(close_1d), np.nan)  # Central Pivot Point
    
    for i in range(len(close_1d)):
        H = high_1d[i]
        L = low_1d[i]
        C = close_1d[i]
        RANGE = H - L
        camarilla_CPP[i] = (H + L + C) / 3
        camarilla_S3[i] = C - (RANGE * 1.1 / 2)
        camarilla_R3[i] = C + (RANGE * 1.1 / 2)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Align 1d Camarilla levels to 12h timeframe
    S3_12h = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    R3_12h = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    CPP_12h = align_htf_to_ltf(prices, df_1d, camarilla_CPP)
    
    # Align 1w EMA50 to 12h timeframe
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Camarilla levels, EMA50, and volume MA20
    start_idx = max(19, 0)  # 19 for volume MA20, others handled by align_htf_to_ltf
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(S3_12h[i]) or np.isnan(R3_12h[i]) or 
            np.isnan(CPP_12h[i]) or np.isnan(ema_1w_12h[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: break above R3 with 1w EMA50 uptrend and volume filter
            if (price > R3_12h[i] and 
                price > ema_1w_12h[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below S3 with 1w EMA50 downtrend and volume filter
            elif (price < S3_12h[i] and 
                  price < ema_1w_12h[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below CPP
            if price < CPP_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above CPP
            if price > CPP_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_S3R3_Breakout_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0