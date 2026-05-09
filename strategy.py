#!/usr/bin/env python3
# 1h_4h_1d_TrendFollowing_Momentum
# Hypothesis: Combines 1d trend filter (close above/below 200 EMA) with 4h momentum (RSI divergence) and 1h entry timing.
# Uses 1d EMA200 for trend direction, 4h RSI for momentum confirmation, and 1h price action for entry.
# Designed to work in both trending and ranging markets by filtering trades with higher timeframe trend.
# Target: 15-30 trades/year per symbol with disciplined risk management.

name = "1h_4h_1d_TrendFollowing_Momentum"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 + ema_200_1d[i-1] * 198) / 200
    
    # Calculate 4h RSI(14) for momentum
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_4h, np.nan)
    avg_loss = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(close_4h)):
            avg_gain[i] = (gain[i] + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i] + avg_loss[i-1] * 13) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Align higher timeframe indicators to 1h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Above 1d EMA200 AND 4h RSI > 50 (bullish momentum)
            if close[i] > ema_200_1d_aligned[i] and rsi_4h_aligned[i] > 50:
                signals[i] = 0.20
                position = 1
            # Enter short: Below 1d EMA200 AND 4h RSI < 50 (bearish momentum)
            elif close[i] < ema_200_1d_aligned[i] and rsi_4h_aligned[i] < 50:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Below 1d EMA200 OR 4h RSI < 40 (loss of momentum)
            if close[i] < ema_200_1d_aligned[i] or rsi_4h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Above 1d EMA200 OR 4h RSI > 60 (loss of momentum)
            if close[i] > ema_200_1d_aligned[i] or rsi_4h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals