#!/usr/bin/env python3
# 1d_Camarilla_R1S1_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot level breaks (R1/S1) on daily chart with weekly trend filter (EMA34) and volume confirmation.
# Works in bull markets (breakouts above R1 in uptrend) and bear markets (breakdowns below S1 in downtrend).
# Target: 15-25 trades/year to minimize fee drag on 1d timeframe.

name = "1d_Camarilla_R1S1_Breakout_1wTrend_Volume"
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
    
    # Calculate daily Camarilla levels (using previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's high-low-close to calculate today's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    valid_idx = ~np.isnan(prev_high) & ~np.isnan(prev_low) & ~np.isnan(prev_close)
    camarilla_r1 = np.full_like(prev_close, np.nan)
    camarilla_s1 = np.full_like(prev_close, np.nan)
    
    camarilla_r1[valid_idx] = prev_close[valid_idx] + 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    camarilla_s1[valid_idx] = prev_close[valid_idx] - 1.1 * (prev_high[valid_idx] - prev_low[valid_idx]) / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation in uptrend (close > weekly EMA34)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               high[i] > camarilla_r1_aligned[i] and volume_confirmed[i] and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume confirmation in downtrend (close < weekly EMA34)
            elif camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
                 low[i] < camarilla_s1_aligned[i] and volume_confirmed[i] and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below R1 or trend weakens (close < weekly EMA34)
            if camarilla_r1_aligned[i] > 0 and not np.isnan(camarilla_r1_aligned[i]) and \
               low[i] < camarilla_r1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above S1 or trend weakens (close > weekly EMA34)
            if camarilla_s1_aligned[i] > 0 and not np.isnan(camarilla_s1_aligned[i]) and \
               high[i] > camarilla_s1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals