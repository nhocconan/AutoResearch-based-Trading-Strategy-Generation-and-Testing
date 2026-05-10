#!/usr/bin/env python3
# 12h_Camarilla_Pivot_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (S3/S4 and R3/R4) from the daily timeframe act as strong support/resistance.
# A breakout beyond these levels with volume confirmation and aligned daily trend (EMA34) signals a sustained move.
# Designed for low trade frequency (15-30/year) to minimize fee drift on 12h chart.

name = "12h_Camarilla_Pivot_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # --- Daily HTF data for Camarilla and trend ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Daily high, low, close for Camarilla
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    range_1d = high_1d - low_1d
    r4_1d = close_1d + 1.5 * range_1d
    r3_1d = close_1d + 1.1 * range_1d
    s3_1d = close_1d - 1.1 * range_1d
    s4_1d = close_1d - 1.5 * range_1d
    
    # Align Camarilla levels to 12h chart (wait for daily close)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Daily EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 12h indicators ---
    # Volume confirmation: 24-period average (2 days of 12h bars)
    close_series = pd.Series(close)
    vol_ma = close_series.rolling(window=24, min_periods=24).mean()  # using close as proxy, will replace
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 35)  # enough for vol_ma and daily indicators
    
    for i in range(start_idx, n):
        if np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or \
           np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above R4 AND daily uptrend (close > EMA34)
            if close[i] > r4_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S4 AND daily downtrend (close < EMA34)
            elif close[i] < s4_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below R3 OR trend turns down
            if close[i] < r3_1d_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above S3 OR trend turns up
            if close[i] > s3_1d_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals