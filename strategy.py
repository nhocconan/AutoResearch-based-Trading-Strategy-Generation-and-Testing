#!/usr/bin/env python3
"""
4h_Bollinger_Bands_Mean_Reversion_With_Trend_Filter
Hypothesis: In ranging markets, price tends to revert from Bollinger Bands extremes.
In trending markets, price tends to stay within the upper/lower band.
Use Bollinger Bands (20,2) on 4h with trend filter from 1d EMA34 and volume confirmation.
Long when price touches lower band in uptrend with volume spike.
Short when price touches upper band in downtrend with volume spike.
Exit when price returns to middle band or trend reverses.
Targets 80-150 trades over 4 years (20-38/year) to balance opportunity and fee cost.
Works in both bull (buy dips to lower band) and bear (sell rallies to upper band).
"""

name = "4h_Bollinger_Bands_Mean_Reversion_With_Trend_Filter"
timeframe = "4h"
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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Bollinger Bands (20,2) on 4h
    bb_period = 20
    bb_std = 2
    sma_bb = np.full(n, np.nan)
    std_bb = np.full(n, np.nan)
    upper_bb = np.full(n, np.nan)
    lower_bb = np.full(n, np.nan)
    middle_bb = np.full(n, np.nan)
    
    if n >= bb_period:
        for i in range(bb_period - 1, n):
            sma_bb[i] = np.mean(close[i - bb_period + 1:i + 1])
            std_bb[i] = np.std(close[i - bb_period + 1:i + 1])
            upper_bb[i] = sma_bb[i] + bb_std * std_bb[i]
            lower_bb[i] = sma_bb[i] - bb_std * std_bb[i]
            middle_bb[i] = sma_bb[i]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 34, 1)  # Need Bollinger Bands, EMA34, and volume
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or np.isnan(middle_bb[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled to 4h)
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h periods in 1d
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Price touches lower Bollinger Band in uptrend with volume confirmation
            if close[i] <= lower_bb[i] and close[i] > ema34_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band in downtrend with volume confirmation
            elif close[i] >= upper_bb[i] and close[i] < ema34_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns to middle band or trend reverses to downtrend
            if close[i] >= middle_bb[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns to middle band or trend reverses to uptrend
            if close[i] <= middle_bb[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals