#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS
Hypothesis: For 12h timeframe, uses weekly EMA50 as trend filter and daily RSI for momentum confirmation, combined with tight Camarilla R1/S1 breakouts and volume spikes. Targets 15-25 trades/year to minimize fee decay while capturing strong institutional moves. Weekly trend filter reduces whipsaws in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for RSI momentum and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily RSI for momentum confirmation
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla formulas: range = (H - L), multiplier = 1.12
    # R1 = C + (H-L)*1.12/12, S1 = C - (H-L)*1.12/12
    rng = (high_1d - low_1d)
    r1 = close_1d_prev + rng * 1.12 / 12
    s1 = close_1d_prev - rng * 1.12 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.8 * 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for weekly EMA, daily RSI, volume MA, and Camarilla
    start_idx = max(50, 14, 30)  # EMA50 weekly, RSI14 daily, VolMA30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema50_1w_aligned[i]
        rsi_val = rsi_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Only trade when price is above weekly EMA50 (uptrend bias)
        if close[i] < weekly_trend:
            # In downtrend, only consider shorts with extreme RSI
            if position == 0 and rsi_val < 30 and close[i] < s1_level and vol_spike_val:
                signals[i] = -size
                position = -1
            elif position == -1 and (close[i] > r1_level or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
        else:
            # Uptrend: look for longs
            if position == 0 and rsi_val > 50 and close[i] > r1_level and vol_spike_val:
                signals[i] = size
                position = 1
            elif position == 1 and (close[i] < s1_level or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size if position == 1 else 0.0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeS"
timeframe = "12h"
leverage = 1.0