#!/usr/bin/env python3
"""
4h_Keltner_Breakout_Pullback
Hypothesis: 4h timeframe with Keltner Channel breakout and EMA pullback entry.
Goes long when price breaks above upper Keltner Channel (EMA20 + 2*ATR) and pulls back to EMA20 in uptrend.
Goes short when price breaks below lower Keltner Channel (EMA20 - 2*ATR) and pulls back to EMA20 in downtrend.
Uses 1-day ADX for trend strength filter (ADX > 25) to avoid ranging markets.
Designed for 20-40 trades/year to avoid fee drag in 4h timeframe.
Works in bull/bear via trend filter and pullback entries at dynamic support/resistance.
"""

name = "4h_Keltner_Breakout_Pullback"
timeframe = "4h"
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
    
    # Get 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily timeframe
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = np.zeros_like(high)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        # Set first 'period' values to NaN
        adx[:period] = np.nan
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Keltner Channel on 4h timeframe
    def calculate_atr(high, low, close, period=10):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Set first 'period' values to NaN
        atr[:period] = np.nan
        return atr
    
    atr_4h = calculate_atr(high, low, close, 10)
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    upper_keltner = ema_20_4h + 2 * atr_4h
    lower_keltner = ema_20_4h - 2 * atr_4h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h[i]) or 
            np.isnan(upper_keltner[i]) or 
            np.isnan(lower_keltner[i]) or
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Keltner and pulls back to EMA20 in uptrend
            if (high[i] > upper_keltner[i] and 
                low[i] <= ema_20_4h[i] and 
                close[i] > ema_20_4h[i] and
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner and pulls back to EMA20 in downtrend
            elif (low[i] < lower_keltner[i] and 
                  high[i] >= ema_20_4h[i] and 
                  close[i] < ema_20_4h[i] and
                  strong_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Keltner or trend weakens
            if low[i] < lower_keltner[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Keltner or trend weakens
            if high[i] > upper_keltner[i] or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals