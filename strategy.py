#!/usr/bin/env python3
# 1H_EMA20_RSI14_4HTrend_1DFilter
# Hypothesis: Enter trades in direction of 4h trend with 1d filter and 1h entry timing.
# Long when: 4h EMA20 > EMA50 (uptrend), 1d close > EMA50, RSI(14) < 40 (pullback), and price near EMA20.
# Short when: 4h EMA20 < EMA50 (downtrend), 1d close < EMA50, RSI(14) > 60 (bounce), and price near EMA20.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 15-35 trades/year per symbol by using higher timeframe for direction and 1h for precise entry.

name = "1H_EMA20_RSI14_4HTrend_1DFilter"
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
    
    # Precompute hour filter
    hours = prices.index.hour
    
    # 1h indicators
    close_s = pd.Series(close)
    
    # EMA20 and EMA50 for 1h trend and dynamic support/resistance
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14)
    delta = close_s.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 4h trend filter: EMA20 vs EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_gt_ema50_4h = ema20_4h > ema50_4h
    ema20_lt_ema50_4h = ema20_4h < ema50_4h
    
    # Align 4h trend to 1h
    ema20_gt_ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_gt_ema50_4h.astype(float))
    ema20_lt_ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_lt_ema50_4h.astype(float))
    
    # 1d trend filter: close vs EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align 1d trend to 1h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = 60
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(rsi[i]) or
            np.isnan(ema20_gt_ema50_4h_aligned[i]) or np.isnan(ema20_lt_ema50_4h_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_near_ema20 = abs(close[i] - ema20[i]) / ema20[i] < 0.015  # within 1.5%
        
        ema20_up_4h = ema20_gt_ema50_4h_aligned[i] > 0.5
        ema20_down_4h = ema20_lt_ema50_4h_aligned[i] > 0.5
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: 4h uptrend + 1d uptrend + RSI oversold + price near EMA20
            if ema20_up_4h and daily_up and rsi[i] < 40 and price_near_ema20:
                signals[i] = 0.20
                position = 1
            # Enter short: 4h downtrend + 1d downtrend + RSI overbought + price near EMA20
            elif ema20_down_4h and daily_down and rsi[i] > 60 and price_near_ema20:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: 4h trend breaks down or 1d trend breaks down or RSI overbought
            if not ema20_up_4h or not daily_up or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: 4h trend breaks up or 1d trend breaks up or RSI oversold
            if not ema20_down_4h or not daily_down or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals