#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_1d_ADX_Filter
Hypothesis: KAMA adapts to market noise, providing a smooth trend filter. 
Trend direction is determined by price relative to KAMA(10). 
Entries are taken only when 1d ADX > 25 (trending market) to avoid whipsaws in ranging periods.
Exit when price crosses back over KAMA or ADX falls below 20.
Target: 20-30 trades/year (80-120 total) to minimize fee drag.
Works in bull/bear by only trading in strong trends.
"""

name = "4h_KAMA_Trend_With_1d_ADX_Filter"
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
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA(10) on 4h close
    er = np.zeros(n)
    change = np.abs(np.diff(close, prepend=close[0]))
    for i in range(10, n):
        direction = np.abs(close[i] - close[i-10])
        volatility = np.sum(change[i-9:i+1])
        er[i] = direction / volatility if volatility != 0 else 0
    sc = (er * 0.59 + 0.06) ** 2
    kama = np.full(n, np.nan)
    kama[9] = np.mean(close[:10])
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # ADX(14) on 1d
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            elif minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], 
                       np.abs(high[i] - close[i-1]), 
                       np.abs(low[i] - close[i-1]))
        # Smooth
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(plus_dm)
        minus_di = np.zeros_like(minus_dm)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_di[period-1] = np.mean(plus_dm[1:period]) / atr[period-1] * 100
            minus_di[period-1] = np.mean(minus_dm[1:period]) / atr[period-1] * 100
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_di[i] = (plus_di[i-1] * (period-1) + plus_dm[i]) / atr[i] * 100
                minus_di[i] = (minus_di[i-1] * (period-1) + minus_dm[i]) / atr[i] * 100
        dx = np.zeros_like(atr)
        dx[:] = np.nan
        mask = (plus_di + minus_di) != 0
        dx[mask] = np.abs(plus_di[mask] - minus_di[mask]) / (plus_di[mask] + minus_di[mask]) * 100
        adx = np.full_like(dx, np.nan)
        if len(high) >= 2*period-1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for KAMA and ADX
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx = adx_1d_aligned[i]
        is_trending = adx > 25
        is_weak_trend = adx < 20
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        if position == 0:
            # Enter long in uptrend with strong ADX
            if price_above_kama and is_trending:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend with strong ADX
            elif price_below_kama and is_trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below KAMA or trend weakens
            if price_below_kama or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above KAMA or trend weakens
            if price_above_kama or is_weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals