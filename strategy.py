#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_With_Volume_Confirmation
Hypothesis: KAMA (Kaufman Adaptive Moving Average) trend filter combined with 1w trend and volume confirmation.
KAMA adapts to market noise, staying close to price in trends and away in ranges.
We go long when KAMA turns up (bullish) and price is above KAMA, short when KAMA turns down and price below KAMA.
1w trend filter ensures alignment with higher timeframe trend. Volume confirmation filters weak signals.
Designed to work in both bull (follow trend) and bear (counter-trend bounces within trend) markets.
Target: 30-100 total trades over 4 years (7-25/year).
"""

name = "1d_KAMA_Trend_Filter_With_Volume_Confirmation"
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
    
    # 1w trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1w volume SMA20 for volume confirmation
    volume_1w = df_1w['volume'].values
    vol_sma20_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(volume_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    # KAMA (Kaufman Adaptive Moving Average) - 14-period ER, 2-30 SC
    er_period = 10
    fast_sc = 2
    slow_sc = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, n=er_period))
    abs_change = np.abs(np.diff(close))
    er = np.full(n, np.nan)
    if n >= er_period + 1:
        sum_abs = np.nansum(abs_change[1:er_period+1])
        er[er_period] = change[0] / sum_abs if sum_abs != 0 else 0
        for i in range(er_period + 1, n):
            sum_abs = sum_abs - abs_change[i-er_period] + abs_change[i-1]
            er[i] = change[i-er_period] / sum_abs if sum_abs != 0 else 0
    
    # Smoothing Constant
    sc = np.full(n, np.nan)
    if n >= er_period + 1:
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA
    kama = np.full(n, np.nan)
    if n >= er_period + 1:
        kama[er_period] = close[er_period]
        for i in range(er_period + 1, n):
            if not np.isnan(sc[i]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = er_period + 1
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average weekly volume (scaled to daily)
        # Approximate daily volume from weekly: weekly volume / 5
        vol_daily_approx = vol_sma20_1w_aligned[i] / 5.0
        volume_confirm = volume[i] > 1.5 * vol_daily_approx
        
        if position == 0:
            # Long: Price above KAMA, KAMA rising, uptrend on 1w, volume confirmation
            if close[i] > kama[i] and kama[i] > kama[i-1] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA, KAMA falling, downtrend on 1w, volume confirmation
            elif close[i] < kama[i] and kama[i] < kama[i-1] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below KAMA or trend reversal
            if close[i] < kama[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or trend reversal
            if close[i] > kama[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals