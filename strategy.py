#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: On 12h timeframe, price breaking above/below Camarilla R1/S1 levels from prior 1d candle acts as momentum signal in trending markets. Uses 1d EMA50 for trend filter and 1d volume spike (>1.5x average) for confirmation. Only enters in direction of 1d trend. Designed for low trade frequency (12-37/year) with clear entry/exit rules to minimize fee drag. Works in bull/bear by following higher timeframe trend.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle first value
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    prev_close[0] = df_1d['close'].iloc[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    cam_pp = (prev_high + prev_low + prev_close) / 3
    cam_r1 = cam_pp + (range_ * 1.1 / 12)
    cam_s1 = cam_pp - (range_ * 1.1 / 12)
    cam_r2 = cam_pp + (range_ * 1.1 / 6)
    cam_s2 = cam_pp - (range_ * 1.1 / 6)
    
    # Align Camarilla levels to 12h
    cam_r1_12h = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_12h = align_htf_to_ltf(prices, df_1d, cam_s1)
    cam_r2_12h = align_htf_to_ltf(prices, df_1d, cam_r2)
    cam_s2_12h = align_htf_to_ltf(prices, df_1d, cam_s2)
    
    # 1d EMA50 for trend filter
    ema50_1d = df_1d['close'].ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average for confirmation
    vol_avg_1d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    vol_avg_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 12h price data
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(cam_r1_12h[i]) or np.isnan(cam_s1_12h[i]) or 
            np.isnan(ema50_12h[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Exit logic: close position if reverse signal or stop hit
                if position == 1 and close_12h[i] <= cam_s1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= cam_r1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        if position == 0:
            # Look for entries in direction of 1d trend
            if vol_confirm:
                # Long: price breaks above R1 and above EMA50 (uptrend)
                if close_12h[i] > cam_r1_12h[i] and close_12h[i] > ema50_12h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_12h[i]
                # Short: price breaks below S1 and below EMA50 (downtrend)
                elif close_12h[i] < cam_s1_12h[i] and close_12h[i] < ema50_12h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long: exit if price breaks below S1 (trend reversal) or at R2 (target)
                if close_12h[i] < cam_s1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif close_12h[i] >= cam_r2_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price breaks above R1 (trend reversal) or at S2 (target)
                if close_12h[i] > cam_r1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif close_12h[i] <= cam_s2_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals