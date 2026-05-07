#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v5
# Hypothesis: Further tighten entry conditions to reduce trade frequency below 25/year by requiring
# consecutive closes beyond R1/S1 (2-bar confirmation), stronger volume spike (3x average),
# and ADX filter to avoid ranging markets. Target: 15-20 trades/year to minimize fee drag.
# Uses 1-day EMA34 trend filter and ADX(14) > 25 for trend strength. Designed to work in both
# bull and bear markets by following the trend on higher timeframe.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeS_v5"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Get 1d data for trend filter (EMA34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ADX(14) on 1d for trend strength filter
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    def WilderSmooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmooth(tr, 14)
    di_plus = WilderSmooth(dm_plus, 14)
    di_minus = WilderSmooth(dm_minus, 14)
    
    # Avoid division by zero
    dx = np.where((di_plus + di_minus) != 0, 
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = WilderSmooth(dx, 14)
    
    # Align all indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike filter on 4h (30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (3.0 * vol_ma_30)  # Increased threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Track consecutive closes for confirmation
    consecutive_high = 0
    consecutive_low = 0
    
    for i in range(60, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(adx_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                consecutive_high = 0
                consecutive_low = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Reset counters
            consecutive_high = 0
            consecutive_low = 0
            
            # Check for consecutive closes beyond levels
            if close[i] > r1_4h[i]:
                consecutive_high += 1
            if close[i] < s1_4h[i]:
                consecutive_low += 1
            
            # Require 2 consecutive closes beyond level + volume spike + strong trend
            if (consecutive_high >= 2 and 
                close[i] > ema_34_1d_4h[i] and 
                volume_spike[i] and 
                adx_4h[i] > 25):
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif (consecutive_low >= 2 and 
                  close[i] < ema_34_1d_4h[i] and 
                  volume_spike[i] and 
                  adx_4h[i] > 25):
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions: require minimum 4 bars held
            if bars_since_entry >= 4:
                if close[i] < r1_4h[i] or close[i] < ema_34_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    consecutive_high = 0
                    consecutive_low = 0
                else:
                    signals[i] = 0.25
            else:
                # Hold position for minimum period
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions: require minimum 4 bars held
            if bars_since_entry >= 4:
                if close[i] > s1_4h[i] or close[i] > ema_34_1d_4h[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                    consecutive_high = 0
                    consecutive_low = 0
                else:
                    signals[i] = -0.25
            else:
                # Hold position for minimum period
                signals[i] = -0.25
    
    return signals