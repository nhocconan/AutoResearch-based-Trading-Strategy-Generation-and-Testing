#!/usr/bin/env python3
# 6h_ElderRay_Rebalance_Strategy
# Hypothesis: Elder Ray Index (bull/bear power) identifies institutional buying/selling pressure.
# In bull markets, we go long when bear power weakens while bull power remains strong.
# In bear markets, we go short when bull power weakens while bear power remains strong.
# Uses 1-day EMA13 as trend filter and volume confirmation to avoid false signals.
# Designed for low trade frequency (15-25/year) to minimize fee drag in ranging/trending markets.

name = "6h_ElderRay_Rebalance_Strategy"
timeframe = "6h"
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
    
    # Get 1-day data for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA13 for trend filter (smooth, lag-appropriate)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13(close)
    # Bear Power = Low - EMA13(close)
    # We need EMA13 of close for the calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Smooth the power values to reduce noise (6-period EMA)
    bull_power_smooth = pd.Series(bull_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Volume confirmation (20-period average on 6h = ~5 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or \
           np.isnan(bear_power_smooth[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirm = volume[i] > 1.3 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: bull power strengthening (>0 and rising) AND bear power weakening (<0 but rising)
            # with volume confirmation and price above daily EMA13 (uptrend bias)
            bull_rising = bull_power_smooth[i] > bull_power_smooth[i-1]
            bear_weakening = bear_power_smooth[i] > bear_power_smooth[i-1]  # less negative
            if bull_power_smooth[i] > 0 and bull_rising and bear_weakening and \
               volume_confirm and close[i] > ema_13_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bear power strengthening (<0 and falling) AND bull power weakening (>0 but falling)
            # with volume confirmation and price below daily EMA13 (downtrend bias)
            elif bear_power_smooth[i] < 0 and bear_power_smooth[i] < bear_power_smooth[i-1] and \
                 bull_power_smooth[i] < bull_power_smooth[i-1] and \
                 volume_confirm and close[i] < ema_13_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bear power turns positive OR bull power turns negative
            if bear_power_smooth[i] >= 0 or bull_power_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bull power turns negative OR bear power turns positive
            if bull_power_smooth[i] >= 0 or bear_power_smooth[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals