#!/usr/bin/env python3
# 6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike
# Hypothesis: Uses Camarilla R3/S3 levels from 12h timeframe. Enters long when price breaks above R3 with volume confirmation and 12h uptrend (close > EMA34).
# Enters short when price breaks below S3 with volume confirmation and 12h downtrend (close < EMA34).
# Exits when price returns to the central pivot point (CP) or reverses direction.
# Uses 12h EMA34 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 12-37 trades per year on 6h timeframe with position size 0.25.

name = "6H_Camarilla_R3_S3_Breakout_12hTrend_VolumeSpike"
timeframe = "6h"
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
    
    # Get 12h data for Camarilla pivots and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend direction
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla pivot levels from previous 12h bar
    # R4 = C + ((H-L)*1.1/2)
    # R3 = C + ((H-L)*1.1/4)
    # R2 = C + ((H-L)*1.1/6)
    # R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12)
    # S2 = C - ((H-L)*1.1/6)
    # S3 = C - ((H-L)*1.1/4)
    # S4 = C - ((H-L)*1.1/2)
    # CP = (H + L + C) / 3
    
    # Shift by 1 to use previous day's data
    prev_high = np.roll(df_12h['high'].values, 1)
    prev_low = np.roll(df_12h['low'].values, 1)
    prev_close = np.roll(df_12h['close'].values, 1)
    prev_high[0] = 0  # first day has no previous day
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla levels
    R3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    S3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    CP = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 6h
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    CP_aligned = align_htf_to_ltf(prices, df_12h, CP)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(CP_aligned[i]) or np.isnan(ema_34_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 12h EMA34
        price_above_ema = close[i] > ema_34_12h_aligned[i]
        price_below_ema = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R3 with volume confirmation and uptrend
            if (close[i] > R3_aligned[i] and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S3 with volume confirmation and downtrend
            elif (close[i] < S3_aligned[i] and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot point or trend reverses
            if (close[i] <= CP_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot point or trend reverses
            if (close[i] >= CP_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals