#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d ADX regime filter and volume confirmation
- Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) measures bull/bear strength
- 1d ADX > 25 defines strong trend regime: only trade in trend direction (long if +DI > -DI, short if -DI > +DI)
- Volume confirmation (> 1.5x 20-period average) filters weak breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by trading with 1d trend regime
- Elder Ray provides timely reversal signals while ADX regime filter avoids choppy markets
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
    
    # Calculate Elder Ray components on 6h
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Higher values = stronger bulls
    bear_power = low - ema_13   # Lower values = stronger bears (more negative)
    
    # Calculate 1d ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], 
                               np.abs(high_1d[1:] - close_1d[:-1])), 
                    np.abs(low_1d[1:] - close_1d[:-1]))
    
    # Handle first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[high_1d[0] - low_1d[0]], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    plus_di_1d = 100 * wilders_smoothing(plus_dm, period) / np.where(atr_1d == 0, 1, atr_1d)
    minus_di_1d = 100 * wilders_smoothing(minus_dm, period) / np.where(atr_1d == 0, 1, atr_1d)
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / np.where((plus_di_1d + minus_di_1d) == 0, 1, (plus_di_1d + minus_di_1d))
    adx_1d = wilders_smoothing(dx_1d, period)
    
    # Align 1d indicators to 6h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    plus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, plus_di_1d)
    minus_di_1d_aligned = align_htf_to_ltf(prices, df_1d, minus_di_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(plus_di_1d_aligned[i]) or np.isnan(minus_di_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power rising, strong uptrend regime, volume spike
            bull_rising = bull_power[i] > bull_power[i-1]
            strong_uptrend = (adx_1d_aligned[i] > 25) and (plus_di_1d_aligned[i] > minus_di_1d_aligned[i])
            volume_spike = volume[i] > 1.5 * vol_ma[i]
            
            # Short conditions: Bear Power falling, strong downtrend regime, volume spike
            bear_falling = bear_power[i] < bear_power[i-1]
            strong_downtrend = (adx_1d_aligned[i] > 25) and (minus_di_1d_aligned[i] > plus_di_1d_aligned[i])
            
            if bull_rising and strong_uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            elif bear_falling and strong_downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Power reversal or trend regime change
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power rising above zero OR trend regime turns bearish
                if (bear_power[i] > 0 and bear_power[i] > bear_power[i-1]) or \
                   (adx_1d_aligned[i] < 20) or \
                   (minus_di_1d_aligned[i] > plus_di_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Bull Power falling below zero OR trend regime turns bullish
                if (bull_power[i] < 0 and bull_power[i] < bull_power[i-1]) or \
                   (adx_1d_aligned[i] < 20) or \
                   (plus_di_1d_aligned[i] > minus_di_1d_aligned[i]):
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