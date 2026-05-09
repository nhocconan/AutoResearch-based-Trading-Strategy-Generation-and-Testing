#!/usr/bin/env python3
# 12h_WeeklyTrend_DailyRSI_Volume
# Hypothesis: Combines weekly trend filter with daily RSI mean-reversion for high-conviction trades.
# In a weekly uptrend (price > weekly EMA), look for long entries when daily RSI is oversold (<30).
# In a weekly downtrend (price < weekly EMA), look for short entries when daily RSI is overbought (>70).
# Volume confirmation on the 12h chart ensures participation. Designed for low trade frequency
# (target: 15-35 trades/year) to minimize fee drag and work in both bull and bear markets.
# Uses mandated 12h timeframe with 1w/1d HTF data as required by experiment #145368.

name = "12h_WeeklyTrend_DailyRSI_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Increased warmup for stability
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1. Get and process Weekly data for trend filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need enough for EMA(20)
        return np.zeros(n)
    
    # Weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    weekly_ema = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # --- 2. Get and process Daily data for entry signal ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough for RSI(14)
        return np.zeros(n)
    
    # Daily RSI(14) for mean-reversion signals
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    daily_rsi = 100 - (100 / (1 + rs))
    daily_rsi_aligned = align_htf_to_ltf(prices, df_1d, daily_rsi)
    
    # --- 3. Volume confirmation on 12h chart ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure all indicators are warmed up
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(daily_rsi_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Weekly uptrend + Daily RSI oversold + Volume spike
            if (close[i] > weekly_ema_aligned[i] and 
                daily_rsi_aligned[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Weekly downtrend + Daily RSI overbought + Volume spike
            elif (close[i] < weekly_ema_aligned[i] and 
                  daily_rsi_aligned[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend fails OR RSI exits oversold
            if (close[i] < weekly_ema_aligned[i]) or (daily_rsi_aligned[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend fails OR RSI exits overbought
            if (close[i] > weekly_ema_aligned[i]) or (daily_rsi_aligned[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals