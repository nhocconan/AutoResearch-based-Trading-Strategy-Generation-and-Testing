#!/usr/bin/env python3
"""
6h Weekly Pivot + Daily Trend + Volume Spike
Hypothesis:
- Uses weekly pivot points (from weekly high/low/close) to identify key support/resistance zones.
- In trending markets (price > daily EMA34), look for long entries at weekly S1/S2 with volume confirmation.
- In ranging markets (price near daily EMA34), fade at weekly R1/R2/S1/S2 with volume confirmation.
- Weekly pivots provide structure; daily EMA34 filters regime; volume spike confirms conviction.
- Designed to work in both bull (trend continuation) and bear (mean reversion at pivots) markets.
- Target: 15-35 trades/year (~60-140 over 4 years) with size 0.25.
"""

name = "6h_WeeklyPivot_DailyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === WEEKLY PIVOTS (from weekly OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Typical price for pivot calculation
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    weekly_high = df_1w['high']
    weekly_low = df_1w['low']
    weekly_close = df_1w['close']
    
    # Pivot point and support/resistance levels
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Align weekly levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1w, pivot.values)
    r1_6h = align_htf_to_ltf(prices, df_1w, r1.values)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1.values)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2.values)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2.values)
    r3_6h = align_htf_to_ltf(prices, df_1w, r3.values)
    s3_6h = align_htf_to_ltf(prices, df_1w, s3.values)
    
    # === DAILY TREND FILTER (EMA34) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Trend regime: above/below daily EMA34
    price_above_ema = close > ema_34_6h
    price_below_ema = close < ema_34_6h
    
    # === VOLUME SPIKE (20-period avg) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(ema_34_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Proximity to weekly levels (within 0.5% tolerance)
        def near_level(price, level):
            return abs(price - level) / level < 0.005
        
        near_s1 = near_level(close[i], s1_6h[i])
        near_s2 = near_level(close[i], s2_6h[i])
        near_r1 = near_level(close[i], r1_6h[i])
        near_r2 = near_level(close[i], r2_6h[i])
        near_r3 = near_level(close[i], r3_6h[i])
        near_s3 = near_level(close[i], s3_6h[i])
        
        if position == 0:
            # LONG ENTRY CONDITIONS
            # 1. Trend continuation: price > EMA34 + near S1/S2 + volume spike
            if (price_above_ema[i] and (near_s1 or near_s2) and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # 2. Mean reversion fade: price < EMA34 + near R1/R2 + volume spike
            elif (price_below_ema[i] and (near_r1 or near_r2) and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            
            # SHORT ENTRY CONDITIONS
            # 1. Trend continuation: price < EMA34 + near R1/R2 + volume spike
            elif (price_below_ema[i] and (near_r1 or near_r2) and vol_spike[i]):
                signals[i] = -0.25
                position = -1
            # 2. Mean reversion fade: price > EMA34 + near S1/S2 + volume spike
            elif (price_above_ema[i] and (near_s1 or near_s2) and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # EXIT: price reaches opposite weekly level or volatility drops
            if near_r1[i] or near_r2[i] or near_r3[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # EXIT: price reaches opposite weekly level or volatility drops
            if near_s1[i] or near_s2[i] or near_s3[i] or not vol_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals