#!/usr/bin/env python3
# 12h_1w_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakouts above weekly Camarilla R1 in uptrend (price > EMA34 on 1d) and breakdowns below S1 in downtrend (price < EMA34 on 1d), with volume confirmation on 12h. Uses weekly context for trend, daily for trend filter, 12h for execution. Designed for low trade frequency to avoid fee drag in both bull and bear markets.

name = "12h_1w_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get weekly data for Camarilla levels (based on previous week's range)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each weekly bar (based on previous week's range)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    camarilla_r1[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    camarilla_s1[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Get daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation in uptrend (price > EMA34 on 1d)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               high[i] > camarilla_r1_aligned[i] and volume_confirmed[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation in downtrend (price < EMA34 on 1d)
            elif camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
                 low[i] < camarilla_s1_aligned[i] and volume_confirmed[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 or trend weakens (price < EMA34 on 1d)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               low[i] < camarilla_r1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 or trend weakens (price > EMA34 on 1d)
            if camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
               high[i] > camarilla_s1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals