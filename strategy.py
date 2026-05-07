#!/usr/bin/env python3
"""
4h_Pivot_Reversal_12hTrend_VolumeConfirm
Hypothesis: Trade reversals at daily Camarilla pivot levels (S1/S2/R1/R2) filtered by 12h EMA50 trend and volume spikes. Works in both bull/bear by fading extremes in ranging markets and catching pullbacks in trends. Target: 25-50 trades/year.
"""

name = "4h_Pivot_Reversal_12hTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # S2 = C - (H - L) * 1.1/6
    
    # We need daily OHLC - use 1d data from mtf_data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1 = close_1d + range_1d * 1.1 / 12
    s1 = close_1d - range_1d * 1.1 / 12
    r2 = close_1d + range_1d * 1.1 / 6
    s2 = close_1d - range_1d * 1.1 / 6
    
    # Align daily levels to 4h timeframe (use previous day's levels)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend determination: price vs 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long setup: price at S1 or S2 with rejection and volume spike
            # Bullish reversal: close above open AND price near support
            bullish_reversal = close[i] > prices['open'].iloc[i]
            near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.005  # Within 0.5%
            near_s2 = abs(close[i] - s2_aligned[i]) / close[i] < 0.005
            
            if ((near_s1 or near_s2) and bullish_reversal and vol_ratio[i] > 2.0):
                # In uptrend, take S2 bounce; in downtrend/ranging, take S1 bounce
                if trend_up or (not trend_down):  # Not in strong downtrend
                    signals[i] = 0.25
                    position = 1
            
            # Short setup: price at R1 or R2 with rejection and volume spike
            # Bearish reversal: close below open AND price near resistance
            bearish_reversal = close[i] < prices['open'].iloc[i]
            near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.005
            near_r2 = abs(close[i] - r2_aligned[i]) / close[i] < 0.005
            
            if ((near_r1 or near_r2) and bearish_reversal and vol_ratio[i] > 2.0):
                # In downtrend, take R2 rejection; in uptrend/ranging, take R1 rejection
                if trend_down or (not trend_up):  # Not in strong uptrend
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price reaches pivot or shows bearish rejection at resistance
            bullish_reversal = close[i] > prices['open'].iloc[i]
            bearish_reversal = close[i] < prices['open'].iloc[i]
            near_r1 = abs(close[i] - r1_aligned[i]) / close[i] < 0.005
            
            if bearish_reversal and near_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches pivot or shows bullish rejection at support
            bullish_reversal = close[i] > prices['open'].iloc[i]
            bearish_reversal = close[i] < prices['open'].iloc[i]
            near_s1 = abs(close[i] - s1_aligned[i]) / close[i] < 0.005
            
            if bullish_reversal and near_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals