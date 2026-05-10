#!/usr/bin/env python3
# 1h_Camarilla_R1S1_Breakout_4hTrend_Volume
# Hypothesis: Camarilla R1/S1 levels on 4h provide strong support/resistance. Price breaking above R1 in a 4h uptrend or below S1 in a 4h downtrend indicates momentum. Volume confirmation filters false breakouts. 1h timeframe used for entry timing only to reduce noise and overtrading. Works in bull markets by riding uptrends and in bear markets by following downtrends. Session filter (08-20 UTC) applied to avoid low-liquidity hours.

name = "1h_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla levels and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla levels (R1, S1)
    # Formula: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_high = df_4h['high'].values
    daily_low = df_4h['low'].values
    daily_close = df_4h['close'].values
    
    camarilla_range = (daily_high - daily_low) * 1.1 / 12
    camarilla_r1 = daily_close + camarilla_range
    camarilla_s1 = daily_close - camarilla_range
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation (24-period MA on 1h = 1 day)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC (precomputed)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA34 (34) and volume MA (24)
    start_idx = max(34, 24)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 4h trend filter
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Volume confirmation (>1.5x MA to balance sensitivity and filtering)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: 4h uptrend + price breaks above 4h Camarilla R1 + volume
            if uptrend and close[i] > camarilla_r1_aligned[i] and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: 4h downtrend + price breaks below 4h Camarilla S1 + volume
            elif downtrend and close[i] < camarilla_s1_aligned[i] and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below R1
            if not uptrend or close[i] < camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks or price re-enters above S1
            if not downtrend or close[i] > camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals