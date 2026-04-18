#!/usr/bin/env python3
"""
4h_1d_TRIX_VolumeSpike_TrendFilter
Hypothesis: On 4h timeframe, use TRIX (15-period) to identify momentum direction, confirmed by volume spikes (>2x 20-period average) and filtered by 1d EMA50 trend. Enter long when TRIX crosses above zero with volume spike and 1d EMA50 uptrend; short when TRIX crosses below zero with volume spike and 1d EMA50 downtrend. Exit when TRIX crosses back through zero or volume drops below average. Uses TRIX's smoothing to reduce whipsaw and volume confirmation to ensure momentum validity. Targets 20-35 trades/year per symbol via strict entry conditions. Works in bull/bear by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50)
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # TRIX (15-period triple EMA) on 4h close
    trix_period = 15
    ema1 = np.full_like(close, np.nan)
    ema2 = np.full_like(close, np.nan)
    ema3 = np.full_like(close, np.nan)
    trix = np.full_like(close, np.nan)
    
    if len(close) >= trix_period:
        # First EMA
        ema1[trix_period - 1] = np.mean(close[:trix_period])
        for i in range(trix_period, len(close)):
            ema1[i] = (close[i] * 2 / (trix_period + 1)) + (ema1[i-1] * (trix_period - 1) / (trix_period + 1))
        
        # Second EMA of EMA1
        ema2[2*trix_period - 2] = np.mean(ema1[trix_period-1:2*trix_period-1])
        for i in range(2*trix_period - 1, len(close)):
            ema2[i] = (ema1[i] * 2 / (trix_period + 1)) + (ema2[i-1] * (trix_period - 1) / (trix_period + 1))
        
        # Third EMA of EMA2
        ema3[3*trix_period - 3] = np.mean(ema2[2*trix_period-2:3*trix_period-2])
        for i in range(3*trix_period - 2, len(close)):
            ema3[i] = (ema2[i] * 2 / (trix_period + 1)) + (ema3[i-1] * (trix_period - 1) / (trix_period + 1))
        
        # TRIX = 100 * (EMA3 - previous EMA3) / previous EMA3
        trix[trix_period:] = 100 * (ema3[trix_period:] - ema3[:-trix_period]) / ema3[:-trix_period]
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(trix_period, vol_period, 3*trix_period-2)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or i == 0):
            signals[i] = 0.0
            continue
        
        # TRIX zero cross
        trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
        trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: TRIX crosses up + volume spike + 1d EMA50 uptrend
            if (trix_cross_up and vol_spike and 
                i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses down + volume spike + 1d EMA50 downtrend
            elif (trix_cross_down and vol_spike and 
                  i > 0 and not np.isnan(ema_1d_aligned[i-1]) and ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses down or volume drops below average
            if trix_cross_down or volume[i] < vol_ma[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses up or volume drops below average
            if trix_cross_up or volume[i] < vol_ma[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_TRIX_VolumeSpike_TrendFilter"
timeframe = "4h"
leverage = 1.0