#!/usr/bin/env python3
name = "6h_Weekly_Pivot_Reversion_v1"
timeframe = "6h"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly Pivot (standard) from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Weekly Pivot S1/R1 levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align weekly levels to 6h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Weekly RSI(14) for overbought/oversold
    delta = pd.Series(df_1w['close']).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_aligned = align_htf_to_ltf(prices, df_1w, rsi_14)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 14, 4)  # Wait for EMA, RSI, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(rsi_14_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Mean reversion long: price below S1, oversold weekly RSI, volume spike, and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            rsi_oversold = rsi_14_aligned[i] < 30
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] < s1_aligned[i] and rsi_oversold and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Mean reversion short: price above R1, overbought weekly RSI, volume spike, and daily downtrend
            elif close[i] > r1_aligned[i] and rsi_14_aligned[i] > 70 and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back above S1 or RSI normalizes
            if close[i] > s1_aligned[i] or rsi_14_aligned[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back below R1 or RSI normalizes
            if close[i] < r1_aligned[i] or rsi_14_aligned[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Weekly Pivot S1/R1 mean reversion with weekly RSI extremes and daily trend
# - Weekly Pivot S1/R1 act as strong support/resistance from prior week
# - Long when price breaks below S1 with weekly RSI <30 (oversold) + volume spike + daily uptrend
# - Short when price breaks above R1 with weekly RSI >70 (overbought) + volume spike + daily downtrend
# - Weekly RSI adds momentum exhaustion filter to avoid catching falling knives
# - Daily trend filter ensures trades align with higher timeframe momentum
# - Volume spike (1.8x average) confirms institutional interest at extremes
# - Exit when price returns to S1/R1 or RSI normalizes (>50 for longs, <50 for shorts)
# - Position size 0.25 targets 50-150 total trades over 4 years (12-37/year)
# - Works in bull markets (buy S1 bounces in uptrend) and bear markets (sell R1 bounces in downtrend)
# - Novel: Weekly Pivot + Weekly RSI + Daily Trend + Volume confirmation on 6h timeframe