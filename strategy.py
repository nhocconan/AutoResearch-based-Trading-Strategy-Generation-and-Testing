#!/usr/bin/env python3
"""
12h Weekly RSI Mean Reversion + Volume Spike + Daily Trend Filter
Hypothesis: Extreme weekly RSI values (overbought/oversold) on 12h timeframe, when combined with 
volume spikes and aligned with daily trend, capture mean-reversion opportunities in both bull 
and bear markets. Weekly RSI provides fewer, higher-quality signals than lower timeframes.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""
name = "12h_WeeklyRSI_MeanReversion_VolumeTrend"
timeframe = "12h"
leverage = 1.0

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
    
    # === Weekly RSI for mean reversion signal ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha=1/period)
    avg_gain = np.zeros_like(weekly_close)
    avg_loss = np.zeros_like(weekly_close)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(weekly_close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to 12h timeframe
    rsi_12h = align_htf_to_ltf(prices, df_1w, rsi)
    
    # === Daily EMA34 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Volume Spike (24-period on 12h, ~2 weeks) ===
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_12h[i]) or np.isnan(ema_34_12h[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI oversold (<30) + volume spike + price above daily EMA34 (uptrend bias)
            if (rsi_12h[i] < 30 and 
                vol_spike[i] and
                close[i] > ema_34_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI overbought (>70) + volume spike + price below daily EMA34 (downtrend bias)
            elif (rsi_12h[i] > 70 and 
                  vol_spike[i] and
                  close[i] < ema_34_12h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI returns to neutral (>50) or reverse signal
            if rsi_12h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI returns to neutral (<50) or reverse signal
            if rsi_12h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals