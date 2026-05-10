#!/usr/bin/env python3
# 1d_1w_Camarilla_R1_S1_Trend_Reversal
# Hypothesis: Uses weekly Camarilla pivot levels to identify key support/resistance.
# Enters long when price breaks above weekly R1 with volume confirmation and weekly uptrend (weekly close > EMA34).
# Enters short when price breaks below weekly S1 with volume confirmation and weekly downtrend (weekly close < EMA34).
# Exits when price returns to the weekly pivot point (CP) or reverses weekly trend.
# Weekly timeframe reduces trade frequency to avoid fee drag while capturing major turns.
# Designed to work in both bull and bear markets by using weekly trend filter.
# Targets 10-30 trades per year on 1d timeframe with position size 0.25.

name = "1d_1w_Camarilla_R1_S1_Trend_Reversal"
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
    
    # Get weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate weekly Camarilla pivot levels from previous week
    # R1 = C + ((H-L)*1.1/12)
    # S1 = C - ((H-L)*1.1/12)
    # CP = (H + L + C) / 3
    
    # Shift by 1 to use previous week's data
    prev_high = np.roll(df_1w['high'].values, 1)
    prev_low = np.roll(df_1w['low'].values, 1)
    prev_close = np.roll(df_1w['close'].values, 1)
    prev_high[0] = 0  # first week has no previous week
    prev_low[0] = 0
    prev_close[0] = 0
    
    # Calculate Camarilla levels
    R1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    S1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    CP = (prev_high + prev_low + prev_close) / 3
    
    # Align weekly Camarilla levels to daily
    R1_aligned = align_htf_to_ltf(prices, df_1w, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1w, S1)
    CP_aligned = align_htf_to_ltf(prices, df_1w, CP)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or np.isnan(CP_aligned[i]) or np.isnan(ema_34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above weekly R1 with volume confirmation and weekly uptrend
            if (close[i] > R1_aligned[i] and 
                volume_confirm[i] and 
                price_above_ema):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below weekly S1 with volume confirmation and weekly downtrend
            elif (close[i] < S1_aligned[i] and 
                  volume_confirm[i] and 
                  price_below_ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to weekly pivot point or weekly trend reverses
            if (close[i] <= CP_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to weekly pivot point or weekly trend reverses
            if (close[i] >= CP_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals