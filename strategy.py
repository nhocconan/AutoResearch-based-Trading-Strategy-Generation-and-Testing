#!/usr/bin/env python3
# 1H_Camarilla_R3_S3_Breakout_4hTrend
# Hypothesis: Buy breakouts above Camarilla R3 and sell short below S3 during strong 4h trends.
# Uses 4h ADX > 25 for trend filter and 4h EMA50 for direction. Entry on 1h Camarilla breakout.
# Works in bull/bear by following 4h trend direction and using Camarilla levels for precise entries.
# Target: 20-40 trades/year per symbol.

name = "1H_Camarilla_R3_S3_Breakout_4hTrend"
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
    
    # Calculate 1h Camarilla levels (based on previous day)
    # For simplicity, we use rolling window of previous 24h (assuming 24h in 1h data)
    # But proper Camarilla uses previous day's OHLC
    # We'll approximate using 24-period rolling high/low/close
    if len(high) < 24:
        return np.zeros(n)
    
    # Calculate daily OHLC from 1h data (group by day)
    # Since we can't use resample, we'll use rolling window approximation
    # For proper implementation, we need to group by day, but we'll use 24-period window
    roll_high = pd.Series(high).rolling(window=24, min_periods=24).max().shift(1).values
    roll_low = pd.Series(low).rolling(window=24, min_periods=24).min().shift(1).values
    roll_close = pd.Series(close).rolling(window=24, min_periods=24).last().shift(1).values
    
    # Camarilla calculations
    R3 = roll_close + (roll_high - roll_low) * 1.1 / 4
    S3 = roll_close - (roll_high - roll_low) * 1.1 / 4
    
    # 4h trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # EMA50 for trend direction
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ADX for trend strength (14-period)
    # +DM and -DM
    plus_dm = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    minus_dm = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    # True Range
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # +DI and -DI
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                     pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                      pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    # Prepend NaN for alignment
    ema50_4h = np.concatenate([np.full(1, np.nan), ema50_4h])
    adx = np.concatenate([np.full(1, np.nan), adx])
    
    # Align 4h indicators to 1h
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(roll_high[i]) or np.isnan(roll_low[i]) or np.isnan(roll_close[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        in_session = (8 <= hours[i] <= 20)
        strong_trend = adx_aligned[i] > 25
        uptrend_4h = close[i] > ema50_4h_aligned[i]  # Using 1h close vs 4h EMA for simplicity
        downtrend_4h = close[i] < ema50_4h_aligned[i]
        
        if position == 0 and in_session:
            # Enter long: 4h uptrend + strong trend + price breaks above R3
            if uptrend_4h and strong_trend and close[i] > R3[i]:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + strong trend + price breaks below S3
            elif downtrend_4h and strong_trend and close[i] < S3[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: trend weakens or price moves below EMA50
            if not strong_trend or close[i] < ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: trend weakens or price moves above EMA50
            if not strong_trend or close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals