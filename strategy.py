#!/usr/bin/env python3
"""
12h_1w_Camarilla_Breakout_Trend_Filter
Hypothesis: 12h close above/below weekly Camarilla R3/S3 levels with 1w ADX(14) trend filter and volume confirmation. Designed for low trade frequency (15-30/year) by requiring strong weekly breakouts, trend alignment (ADX>25), and volume surge (2x avg). Works in bull/bear via ADX trend filter and mean-reversion exit at weekly pivot. Targets 12h timeframe to reduce overtrading while capturing multi-day trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_Camarilla_Breakout_Trend_Filter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot calculation
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla levels (R3/S3 for breakouts)
    r3_1w = close_1w + range_1w * 1.25
    s3_1w = close_1w - range_1w * 1.25
    
    # Weekly ADX(14) for trend filter
    def calculate_adx(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
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
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Smooth TR, +DM, -DM
        atr = np.zeros_like(high)
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        if len(high) >= period:
            atr[period-1] = np.mean(tr[1:period])
            plus_dm_smooth[period-1] = np.mean(plus_dm[1:period])
            minus_dm_smooth[period-1] = np.mean(minus_dm[1:period])
            for i in range(period, len(high)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
                minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        # Avoid division by zero
        plus_di = np.where(atr != 0, plus_dm_smooth / atr * 100, 0)
        minus_di = np.where(atr != 0, minus_dm_smooth / atr * 100, 0)
        dx = np.where((plus_di + minus_di) != 0, abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        adx = np.full_like(high, np.nan)
        if len(high) >= 2 * period - 1:
            adx[2*period-2] = np.mean(dx[period-1:2*period-1])
            for i in range(2*period-1, len(high)):
                adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    
    # Align weekly data to 12h timeframe
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Volume average (20-period for 12h = ~10 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        trending = adx_1w_aligned[i] > 25
        
        # Volume confirmation: at least 2.0x average
        vol_confirm = volume[i] > 2.0 * vol_avg[i]
        
        # Breakout entries at weekly S3/R3 with trend and volume filters
        long_setup = (close[i] > r3_1w_aligned[i]) and trending and vol_confirm
        short_setup = (close[i] < s3_1w_aligned[i]) and trending and vol_confirm
        
        # Exit when price returns to weekly pivot (mean reversion)
        exit_long = close[i] < pivot_1w_aligned[i]
        exit_short = close[i] > pivot_1w_aligned[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals