#!/usr/bin/env python3
# 12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
# Hypothesis: Price breaking Camarilla R3/S3 levels on 12h chart with 1-week trend filter and volume spike.
# Camarilla levels act as strong support/resistance; breakouts with volume and higher timeframe trend
# capture sustained moves. Weekly trend filter ensures alignment with dominant market direction.
# Volume spike confirms institutional participation. Designed for low trade frequency to minimize fee drag.
# Targets 15-30 trades per year on 12h timeframe.

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly EMA for trend filter (34-period)
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.166/2, R1 = C + (H-L)*1.083/2
    #          S1 = C - (H-L)*1.083/2, S2 = C - (H-L)*1.166/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # where C = (H+L+CLOSE)/3 of previous day
    # We need previous day's OHLC, so we'll calculate for each day and align to 12h bars
    
    # Calculate daily typical price and range for Camarilla
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla R3 and S3 levels
    r3 = typical_price + range_hl * 1.25 / 2
    s3 = typical_price - range_hl * 1.25 / 2
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation (24-period MA on 12h chart ≈ 12 days)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need weekly EMA (34), volume MA (24), and Camarilla (need at least 1 day)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Breakout conditions
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 + weekly uptrend + volume spike
            if breakout_long and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 + weekly downtrend + volume spike
            elif breakout_short and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks back below R3 or weekly trend turns down
            if close[i] < r3_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks back above S3 or weekly trend turns up
            if close[i] > s3_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals