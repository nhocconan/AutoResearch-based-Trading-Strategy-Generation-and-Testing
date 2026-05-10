#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Uses 4h timeframe for trend and price context, 1h for precise entry timing. 
# Enters long when price breaks above 4h R1 with volume confirmation and 4h uptrend (close > EMA34).
# Enters short when price breaks below 4h S1 with volume confirmation and 4h downtrend (close < EMA34).
# Exits when price returns to the 4h pivot point (CP) or reverses direction.
# Uses 4h EMA34 for trend to reduce whipsaws and works in both bull/bear markets.
# Targets 15-37 trades per year on 1h timeframe with position size 0.20.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivots and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend direction
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    prev_high = np.roll(df_4h['high'].values, 1)
    prev_low = np.roll(df_4h['low'].values, 1)
    prev_close = np.roll(df_4h['close'].values, 1)
    prev_high[0] = 0
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla levels
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    CP = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 1h
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    CP_aligned = align_htf_to_ltf(prices, df_4h, CP)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(CP_aligned[i]) or np.isnan(ema_34_4h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = close[i] > ema_34_4h_aligned[i]
        price_below_ema = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 with volume confirmation and uptrend
            if (close[i] > R1_aligned[i] and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 with volume confirmation and downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot point or trend reverses
            if (close[i] <= CP_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to pivot point or trend reverses
            if (close[i] >= CP_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals