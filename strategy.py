#!/usr/bin/env python3
"""
1h_4d_1d_TRIX_Momentum_Trend
Hypothesis: TRIX (12) on 4h identifies momentum shifts; confirmed by 1d EMA(50) trend and 1h RSI(14) pullback.
Enters on TRIX zero-cross with trend alignment during 08-20 UTC session. Avoids whipsaws in sideways markets.
Target: 20-40 trades/year per symbol. Works in bull/bear via trend filter.
"""

name = "1h_4d_1d_TRIX_Momentum_Trend"
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
    
    # Pre-calculate hours for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Get 4h data for TRIX (12)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate TRIX(12): EMA(EMA(EMA(close,12),12),12) then % change
    ema1 = pd.Series(close_4h).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix_raw = ema3.pct_change() * 100  # percentage
    trix = trix_raw.fillna(0).values
    
    # Get 1d data for trend filter (EMA 50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align 4h TRIX and 1d trend to 1h
    trix_aligned = align_htf_to_ltf(prices, df_4h, trix)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    # Calculate 1h RSI(14) for pullback entry
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # warmup for TRIX and EMA
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        trix_now = trix_aligned[i]
        trix_prev = trix_aligned[i-1]
        uptrend = uptrend_1d_aligned[i]
        downtrend = downtrend_1d_aligned[i]
        rsi_now = rsi[i]
        
        if position == 0:
            # LONG: TRIX crosses above zero + 1d uptrend + RSI < 60 (not overbought)
            if trix_prev <= 0 and trix_now > 0 and uptrend and rsi_now < 60:
                signals[i] = 0.20
                position = 1
            # SHORT: TRIX crosses below zero + 1d downtrend + RSI > 40 (not oversold)
            elif trix_prev >= 0 and trix_now < 0 and downtrend and rsi_now > 40:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: TRIX crosses below zero or 1d trend turns down
            if trix_prev >= 0 and trix_now < 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: TRIX crosses above zero or 1d trend turns up
            if trix_prev <= 0 and trix_now > 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals