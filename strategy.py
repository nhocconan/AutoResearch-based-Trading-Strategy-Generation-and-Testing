#!/usr/bin/env python3
"""
1d_ElderRay_WeeklyTrend_Volume
Hypothesis: Elder Ray (Bull/Bear Power) on 1d with 1-week EMA13 trend filter and volume confirmation.
In trending markets, strong bull/bear power persists; in ranging markets, it fades.
Volume filters weak breakouts. Works in bull (strong bull power) and bear (strong bear power).
Target: 30-100 total trades over 4 years (7-25/year).
"""

name = "1d_ElderRay_WeeklyTrend_Volume"
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
    
    # Weekly EMA13 for trend filter (Elder Ray uses EMA13)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema13_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 13:
        ema13_1w[12] = np.mean(close_1w[:13])
        alpha = 2 / (13 + 1)
        for i in range(13, len(close_1w)):
            ema13_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema13_1w[i-1]
    ema13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema13_1w)
    
    # Weekly volume SMA20 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    # Calculate daily EMA13 for Elder Ray (on 1d data)
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        alpha13 = 2 / (13 + 1)
        for i in range(13, n):
            ema13[i] = alpha13 * close[i] + (1 - alpha13) * ema13[i-1]
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # warmup for EMA calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema13_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]) or np.isnan(ema13[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x average weekly volume (scaled to daily)
        # Approximate 1d volume from 1w: 1w volume / 5 (since 5 trading days per week)
        vol_1d_approx = vol_sma20_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.3 * vol_1d_approx
        
        if position == 0:
            # Long: Strong bull power (> 0) with uptrend and volume confirmation
            if bull_power[i] > 0 and close[i] > ema13[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power (< 0) with downtrend and volume confirmation
            elif bear_power[i] < 0 and close[i] < ema13[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bear power becomes negative (bull power fading) or trend reversal
            if bear_power[i] < 0 or close[i] < ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull power becomes positive (bear power fading) or trend reversal
            if bull_power[i] > 0 or close[i] > ema13[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals