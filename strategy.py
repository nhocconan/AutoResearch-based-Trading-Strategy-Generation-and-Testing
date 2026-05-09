#!/usr/bin/env python3
# 1D_1W_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: On daily timeframe, enter long when price breaks above weekly Camarilla R1 level with weekly uptrend and volume confirmation.
# Short when price breaks below weekly Camarilla S1 level with weekly downtrend and volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades and weekly Camarilla levels for key levels.
# Target: 10-25 trades/year per symbol (40-100 total over 4 years) to minimize fee drag.

name = "1D_1W_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for Camarilla levels and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla levels: R1, S1 based on previous week
    typical_price = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    camarilla_r1 = close_1w + (range_1w * 1.1 / 12)
    camarilla_s1 = close_1w - (range_1w * 1.1 / 12)
    
    # Weekly trend: EMA(34) on close
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_1w > ema_34
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align weekly indicators to daily
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or np.isnan(trend_up_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Camarilla R1 + weekly uptrend + volume confirmation
            if close[i] > camarilla_r1_aligned[i] and trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Camarilla S1 + weekly downtrend + volume confirmation
            elif close[i] < camarilla_s1_aligned[i] and not trend_up_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Camarilla S1 (reversal) or trend changes
            if close[i] < camarilla_s1_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Camarilla R1 (reversal) or trend changes
            if close[i] > camarilla_r1_aligned[i] or trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals