#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with 1w EMA50 Trend Filter and Volume Spike Confirmation.
- Long: Price touches Camarilla S3/S4 AND closes above S3 AND price > 1w EMA50 AND volume > 2.0x avg
- Short: Price touches Camarilla R3/R4 AND closes below R3 AND price < 1w EMA50 AND volume > 2.0x avg
- Exit: Opposite Camarilla touch OR price crosses 1w EMA50 (trend flip)
- Uses 1w HTF for EMA50 and Camarilla levels (calculated from prior 1w bar)
- Designed for low trade frequency (12-37/year) to minimize fee drag
- Works in bull (buy reversals at S3/S4 in uptrend) and bear (sell reversals at R3/R4 in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 2.0x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 1w EMA50 for trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Camarilla levels from prior 1w bar (HTF = 1w)
    # Camarilla: H3 = close + 1.25*(high-low), L3 = close - 1.25*(high-low)
    #          H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    camarilla_h3 = close_1w_arr + 1.25 * (high_1w - low_1w)
    camarilla_l3 = close_1w_arr - 1.25 * (high_1w - low_1w)
    camarilla_h4 = close_1w_arr + 1.5 * (high_1w - low_1w)
    camarilla_l4 = close_1w_arr - 1.5 * (high_1w - low_1w)
    
    # Align Camarilla levels to 6h timeframe (use prior completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # Need 50 for EMA, 24 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or
            np.isnan(camarilla_l4_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Camarilla touch conditions (using current bar's high/low vs levels)
        touch_s3_s4 = low[i] <= camarilla_l3_aligned[i] or low[i] <= camarilla_l4_aligned[i]
        touch_r3_r4 = high[i] >= camarilla_h3_aligned[i] or high[i] >= camarilla_h4_aligned[i]
        
        # Camarilla close conditions (using current close vs levels)
        close_above_s3 = close[i] > camarilla_l3_aligned[i]
        close_below_r3 = close[i] < camarilla_h3_aligned[i]
        
        if position == 0:
            # Long: Touch S3/S4 AND close above S3 AND price > 1w EMA50 AND volume confirmation
            if touch_s3_s4 and close_above_s3 and volume_confirm and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Touch R3/R4 AND close below R3 AND price < 1w EMA50 AND volume confirmation
            elif touch_r3_r4 and close_below_r3 and volume_confirm and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Touch R3/R4 OR price < 1w EMA50 (trend flip)
            if touch_r3_r4 or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Touch S3/S4 OR price > 1w EMA50 (trend flip)
            if touch_s3_s4 or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_S3R3_Reversal_1wEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0