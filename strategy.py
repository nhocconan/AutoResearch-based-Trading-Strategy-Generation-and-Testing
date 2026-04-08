#!/usr/bin/env python3
# 6h_adx_cci_trend_follow_v1
# Hypothesis: Trend-following strategy using ADX for trend strength and CCI for momentum entry/exit on 6h timeframe.
# Long when: ADX > 25 (strong trend) and CCI crosses above +100 (bullish momentum).
# Short when: ADX > 25 (strong trend) and CCI crosses below -100 (bearish momentum).
# Exit when CCI crosses back through zero (momentum fade) or ADX drops below 20 (trend weakening).
# Uses 2-3 conditions to avoid overtrading. Target: 15-30 trades/year per symbol.
# Works in both bull/bear markets by following established trends with momentum confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_cci_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ADX calculation (14-period)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First TR is just high-low
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        def smooth_series(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Wilder's smoothing: today = (yesterday * (period-1) + today) / period
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
                else:
                    result[i] = np.nan
            return result
        
        atr = smooth_series(tr, period)
        dm_plus_smooth = smooth_series(dm_plus, period)
        dm_minus_smooth = smooth_series(dm_minus, period)
        
        # Directional Indicators
        di_plus = np.full_like(atr, np.nan)
        di_minus = np.full_like(atr, np.nan)
        dx = np.full_like(atr, np.nan)
        
        for i in range(len(atr)):
            if not np.isnan(atr[i]) and atr[i] != 0:
                di_plus[i] = 100 * dm_plus_smooth[i] / atr[i]
                di_minus[i] = 100 * dm_minus_smooth[i] / atr[i]
                if di_plus[i] + di_minus[i] != 0:
                    dx[i] = 100 * np.abs(di_plus[i] - di_minus[i]) / (di_plus[i] + di_minus[i])
        
        # ADX is smoothed DX
        adx = smooth_series(dx, period)
        return adx
    
    # CCI calculation (20-period)
    def calculate_cci(high, low, close, period=20):
        tp = (high + low + close) / 3
        sma_tp = np.full_like(tp, np.nan)
        mad = np.full_like(tp, np.nan)
        
        for i in range(len(tp)):
            if i >= period - 1:
                sma_tp[i] = np.mean(tp[i-period+1:i+1])
                mad[i] = np.mean(np.abs(tp[i-period+1:i+1] - sma_tp[i]))
        
        cci = np.full_like(tp, np.nan)
        for i in range(len(tp)):
            if not np.isnan(sma_tp[i]) and not np.isnan(mad[i]) and mad[i] != 0:
                cci[i] = (tp[i] - sma_tp[i]) / (0.015 * mad[i])
        return cci
    
    # Calculate indicators
    adx = calculate_adx(high, low, close, 14)
    cci = calculate_cci(high, low, close, 20)
    
    # Get 1d data for trend filter (optional - using 6h ADX as primary trend filter)
    # We'll use 6h ADX as our main trend strength indicator
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 34  # Need enough data for ADX/CCI calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx[i]) or np.isnan(cci[i]) or 
            np.isnan(cci[i-1])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: CCI crosses below zero (momentum fade) or ADX < 20 (weak trend)
            if cci[i] < 0 and cci[i-1] >= 0:
                position = 0
                signals[i] = 0.0
            elif adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: CCI crosses above zero (momentum fade) or ADX < 20 (weak trend)
            if cci[i] > 0 and cci[i-1] <= 0:
                position = 0
                signals[i] = 0.0
            elif adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: ADX > 25 (strong trend) and CCI crosses above +100
            if adx[i] > 25 and cci[i] > 100 and cci[i-1] <= 100:
                position = 1
                signals[i] = 0.25
            # Short entry: ADX > 25 (strong trend) and CCI crosses below -100
            elif adx[i] > 25 and cci[i] < -100 and cci[i-1] >= -100:
                position = -1
                signals[i] = -0.25
    
    return signals