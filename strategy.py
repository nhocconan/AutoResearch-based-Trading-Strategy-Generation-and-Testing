#!/usr/bin/env python3
"""
4h_CCI_Trend_1dFilter_Volume
Hypothesis: CCI(20) identifies cyclical overbought/oversold conditions. In trending markets,
price tends to pull back to the 20-period EMA during pullbacks. We go long when CCI crosses
above -100 from below (end of pullback) in an uptrend (price > EMA50), and short when CCI
crosses below +100 from above (end of bounce) in a downtrend (price < EMA50). The 1d EMA50
filter ensures we only trade in the direction of the higher timeframe trend. Volume
confirmation (current volume > 1.5x 20-period average) filters weak signals. Designed to
work in both bull (buy pullbacks in uptrend) and bear (sell bounces in downtrend) markets.
Target: 20-50 total trades over 4 years (5-12/year).
"""

name = "4h_CCI_Trend_1dFilter_Volume"
timeframe = "4h"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4-period volume average for confirmation (20-period on 1d scaled to 4h)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_avg_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_avg_1d[i] = (vol_avg_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # CCI(20) calculation
    typical_price = (high + low + close) / 3.0
    tp_mean = np.full(n, np.nan)
    tp_dev = np.full(n, np.nan)
    cci = np.full(n, np.nan)
    
    if n >= 20:
        # Initialize first TP mean and mean deviation
        tp_mean[19] = np.mean(typical_price[:20])
        # Mean deviation: average of absolute deviations from mean
        abs_dev = np.abs(typical_price[:20] - tp_mean[19])
        tp_dev[19] = np.mean(abs_dev)
        
        # Recursive calculation for efficiency
        alpha = 1 / 20.0
        for i in range(20, n):
            tp_mean[i] = alpha * typical_price[i] + (1 - alpha) * tp_mean[i-1]
            # Update mean deviation: |price - new_mean| contributes to deviation
            abs_dev_i = np.abs(typical_price[i] - tp_mean[i])
            tp_dev[i] = alpha * abs_dev_i + (1 - alpha) * tp_dev[i-1]
        
        # CCI = (Typical Price - SMA) / (0.015 * Mean Deviation)
        # Avoid division by zero
        valid = tp_dev != 0
        cci[valid] = (typical_price[valid] - tp_mean[valid]) / (0.015 * tp_dev[valid])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # warmup for CCI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(cci[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average 4h volume
        # Approximate 4h volume average from 1d: 1d average / 6 (24h/4h = 6)
        vol_4h_avg = vol_avg_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_4h_avg
        
        if position == 0:
            # Long: CCI crosses above -100 (end of pullback) in uptrend with volume
            if cci[i] > -100 and cci[i-1] <= -100 and close[i] > ema50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below +100 (end of bounce) in downtrend with volume
            elif cci[i] < 100 and cci[i-1] >= 100 and close[i] < ema50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: CCI crosses above +100 (overbought) or trend breaks
            if cci[i] >= 100 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: CCI crosses below -100 (oversold) or trend breaks
            if cci[i] <= -100 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals