#!/usr/bin/env python3
# 4h_Vortex_Trend_With_Volume_Spike
# Hypothesis: Uses Vortex indicator (VI+) and (VI-) to detect trend direction on 4h timeframe.
# Long when VI+ > VI- (bullish trend) with volume confirmation (volume > 2x 20-period average).
# Short when VI- > VI+ (bearish trend) with volume confirmation.
# Includes volatility filter using ATR: only trade when ATR(14) > ATR(50) (volatility expansion).
# Exit when trend reverses (VI+ < VI- for long, VI- < VI+ for short) or volume drops below average.
# Designed to work in both bull and bear markets by following strong trending moves with volume confirmation.
# Target: 20-40 trades/year per symbol with disciplined risk management.

name = "4h_Vortex_Trend_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate True Range components
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan  # First value has no previous close
    tr2[0] = np.nan
    tr3[0] = np.nan
    
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate Vortex Indicator components
    vm_plus = np.abs(high - np.roll(low, 1))
    vm_minus = np.abs(low - np.roll(high, 1))
    vm_plus[0] = np.nan
    vm_minus[0] = np.nan
    
    # Smooth using Wilder's smoothing (similar to RSI/Wilder MA)
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # Initial value: simple average
        result[period-1] = np.nanmean(data[:period])
        # Wilder smoothing: (prev * (period-1) + current) / period
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    vm_plus_smooth = WilderSmooth(vm_plus, period)
    vm_minus_smooth = WilderSmooth(vm_minus, period)
    tr_smooth = WilderSmooth(tr, period)
    
    # Calculate VI+ and VI-
    vi_plus = np.divide(vm_plus_smooth, tr_smooth, out=np.full_like(tr_smooth, np.nan), where=tr_smooth!=0)
    vi_minus = np.divide(vm_minus_smooth, tr_smooth, out=np.full_like(tr_smooth, np.nan), where=tr_smooth!=0)
    
    # Calculate ATR for volatility filter
    atr = tr_smooth  # Already smoothed TR is ATR
    
    # Long-term ATR for volatility regime filter (50-period)
    def WilderSmooth_long(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_long = WilderSmooth_long(tr, 50)
    
    # Volume filter: 20-period average
    def WilderSmooth_vol(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(data[i]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    vol_ma = WilderSmooth_vol(volume, 20)
    volume_ratio = np.divide(volume, vol_ma, out=np.full_like(volume, np.nan), where=vol_ma!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need ATR long and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_long[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when current ATR > long-term ATR (volatility expansion)
        vol_expansion = atr[i] > atr_long[i]
        
        if position == 0:
            # Enter long: VI+ > VI- (bullish trend) + volume confirmation + volatility expansion
            if vi_plus[i] > vi_minus[i] and volume_ratio[i] > 2.0 and vol_expansion:
                signals[i] = 0.25
                position = 1
            # Enter short: VI- > VI+ (bearish trend) + volume confirmation + volatility expansion
            elif vi_minus[i] > vi_plus[i] and volume_ratio[i] > 2.0 and vol_expansion:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend turns bearish (VI- > VI+) or volume drops below average
            if vi_minus[i] > vi_plus[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend turns bullish (VI+ > VI-) or volume drops below average
            if vi_plus[i] > vi_minus[i] or volume_ratio[i] < 1.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals