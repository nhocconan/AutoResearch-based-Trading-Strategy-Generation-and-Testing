#!/usr/bin/env python3
"""
1h_4h1d_RSI_Trend_Pullback
Hypothesis: In strong 4h/1d trends, pullbacks to the 21 EMA on 1h with RSI < 40 (long) or > 60 (short) offer high-probability entries.
Uses volume confirmation (>1.5x average) and session filter (08-20 UTC) to reduce noise.
Targets 20-40 trades/year (80-160 total over 4 years) to minimize fee drift.
Works in bull (buy pullbacks in uptrend) and bear (sell rallies in downtrend).
"""

name = "1h_4h1d_RSI_Trend_Pullback"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # Get 1d data for stronger trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1d EMA21 for trend
    ema_21_1d = pd.Series(df_1d['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    
    # Calculate 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation (20-period MA on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1h RSI (14) and volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_21_4h_aligned[i]) or 
            np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Higher timeframe trend filter (both 4h and 1d must agree)
        uptrend_4h = close[i] > ema_21_4h_aligned[i]
        uptrend_1d = close[i] > ema_21_1d_aligned[i]
        downtrend_4h = close[i] < ema_21_4h_aligned[i]
        downtrend_1d = close[i] < ema_21_1d_aligned[i]
        
        uptrend = uptrend_4h and uptrend_1d
        downtrend = downtrend_4h and downtrend_1d
        
        # Volume confirmation (>1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Session filter (08-20 UTC)
        session_ok = 8 <= hours[i] <= 20
        
        if position == 0:
            # Long entry: uptrend + RSI < 40 (pullback) + volume + session
            if uptrend and rsi_values[i] < 40 and volume_confirm and session_ok:
                signals[i] = 0.20
                position = 1
            # Short entry: downtrend + RSI > 60 (pullback) + volume + session
            elif downtrend and rsi_values[i] > 60 and volume_confirm and session_ok:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or RSI > 60 (overbought)
            if not uptrend or rsi_values[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or RSI < 40 (oversold)
            if not downtrend or rsi_values[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals