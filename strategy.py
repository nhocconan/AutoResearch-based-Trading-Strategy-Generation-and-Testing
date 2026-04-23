#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
- Elder Ray calculates bull power (high - EMA13) and bear power (low - EMA13) to measure buying/selling pressure
- 1d ADX > 25 defines trending regime: only trade when market is trending (avoids choppy losses)
- Volume confirmation (> 1.5x 20-period average) ensures breakout validity
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend regime
- Elder Ray provides institutional-grade pressure measurement with proven edge in trending markets
"""

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
    
    # Calculate 1d ADX for trend regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    
    # Handle first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilders_smoothing(tr, period)
        plus_di = 100 * wilders_smoothing(plus_dm, period) / (atr + 1e-10)
        minus_di = 100 * wilders_smoothing(minus_dm, period) / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = wilders_smoothing(dx, period)
    else:
        adx = np.full_like(close_1d, np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 13-period EMA for Elder Ray (on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # for ADX (14+14+14=42 approx) and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bull power positive, ADX > 25 (trending), and volume confirmation
            long_signal = (bull_power[i] > 0 and 
                          adx_aligned[i] > 25 and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: bear power negative, ADX > 25 (trending), and volume confirmation
            short_signal = (bear_power[i] < 0 and 
                           adx_aligned[i] > 25 and
                           volume[i] > 1.5 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: power divergence or ADX weakening
            exit_signal = False
            
            if position == 1:
                # Exit long: bull power turns negative or ADX < 20 (trend weakening)
                if (bull_power[i] < 0 or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            elif position == -1:
                # Exit short: bear power turns positive or ADX < 20 (trend weakening)
                if (bear_power[i] > 0 or 
                    adx_aligned[i] < 20):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_VolumeConfirm"
timeframe = "6h"
leverage = 1.0