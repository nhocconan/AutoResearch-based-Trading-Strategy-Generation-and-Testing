#!/usr/bin/env python3
# 1d_1w_Camarilla_R3S3_Breakout_TrendFilter
# Hypothesis: Breakouts from weekly Camarilla R3/S3 levels on 1d chart with 1w EMA trend filter and volume spike.
# Camarilla levels provide institutional support/resistance. Breakouts with volume confirm institutional interest.
# Weekly EMA filter ensures trading in direction of higher timeframe trend. Targets 10-25 trades/year to minimize fee drag.
# Works in bull/bear by aligning with weekly trend direction.

name = "1d_1w_Camarilla_R3S3_Breakout_TrendFilter"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for previous week
    # R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Using previous week's OHLC to avoid look-ahead
    wk_high = df_1w['high'].values
    wk_low = df_1w['low'].values
    wk_close = df_1w['close'].values
    
    camarilla_r3 = wk_close + (wk_high - wk_low) * 1.1 / 2
    camarilla_s3 = wk_close - (wk_high - wk_low) * 1.1 / 2
    
    # Align weekly levels to daily timeframe (available after weekly bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    # Weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (20-period MA on daily)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Warmup for volume MA and weekly EMA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation (1.5x average volume)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: price breaks above weekly R3 with uptrend and volume spike
            if close[i] > r3_aligned[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S3 with downtrend and volume spike
            elif close[i] < s3_aligned[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below weekly R3 or trend reverses
            if close[i] < r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above weekly S3 or trend reverses
            if close[i] > s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals