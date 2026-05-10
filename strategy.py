#!/usr/bin/env python3
# 12H_Camarilla_R3_S3_Breakout_1wTrend_1dVolumeSpike
# Hypothesis: Uses Camarilla pivot levels (R3/S3) from 1-day data, filtered by weekly trend (price above/below weekly EMA20) and daily volume spike (volume > 1.5x 20-period average).
# Enters long when price breaks above R3 with weekly uptrend and volume spike.
# Enters short when price breaks below S3 with weekly downtrend and volume spike.
# Exits when price returns to the Camarilla pivot level (close crosses P) or weekly trend reverses.
# Targets 12-37 trades per year on 12h timeframe with position size 0.25.

name = "12H_Camarilla_R3_S3_Breakout_1wTrend_1dVolumeSpike"
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
    
    # Get 1d data for Camarilla pivots and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    vol_series = pd.Series(df_1d['volume'])
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Camarilla: 
    #   H = high, L = low, C = close
    #   R4 = C + (H-L)*1.1/2
    #   R3 = C + (H-L)*1.1/4
    #   S3 = C - (H-L)*1.1/4
    #   P = (H + L + C) / 3
    h = df_1d['high'].values
    l = df_1d['low'].values
    c = df_1d['close'].values
    r3 = c + (h - l) * 1.1 / 4
    s3 = c - (h - l) * 1.1 / 4
    p = (h + l + c) / 3
    
    # Get 1w data for trend (EMA20)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 1d data to 12h
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    p_12h = align_htf_to_ltf(prices, df_1d, p)
    vol_avg_20_12h = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema_20_1w_12h = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup
    
    for i in range(start_idx, n):
        if np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or np.isnan(p_12h[i]) or \
           np.isnan(vol_avg_20_12h[i]) or np.isnan(ema_20_1w_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter: price above/below weekly EMA20
        price_above_ema = close[i] > ema_20_1w_12h[i]
        price_below_ema = close[i] < ema_20_1w_12h[i]
        
        # Daily volume spike: current 12h bar's volume contribution (approximated)
        # Since we don't have intraday volume breakdown, we use the condition that
        # the 12h bar is part of a 1d bar with elevated volume
        # We check if the 1d bar's volume is above its 20-period average
        vol_spike = volume[i] > vol_avg_20_12h[i] * 1.5 if not np.isnan(vol_avg_20_12h[i]) else False
        
        if position == 0:
            # Long entry: price breaks above R3 with weekly uptrend and volume spike
            if (close[i] > r3_12h[i] and 
                price_above_ema and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with weekly downtrend and volume spike
            elif (close[i] < s3_12h[i] and 
                  price_below_ema and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot level (close crosses P) or weekly trend reverses
            if (close[i] < p_12h[i] or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot level (close crosses P) or weekly trend reverses
            if (close[i] > p_12h[i] or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals