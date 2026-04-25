#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dRegimeTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1d trend regime filter (ADX > 25) and volume confirmation (1.5x 20-bar avg). In trending markets, breakout continuation works well. Volume confirms breakout validity. Designed for 4h timeframe targeting 20-40 trades/year. Works in bull/bear by following the trend.
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
    
    # Get 1d data for HTF regime filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14) on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        def Wilder_smoothing(data, period):
            result = np.zeros_like(data)
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
            return result
        
        if len(high) < period + 1:
            return np.full_like(high, np.nan), np.full_like(high, np.nan), np.full_like(high, np.nan)
        
        smoothed_plus_dm = Wilder_smoothing(plus_dm, period)
        smoothed_minus_dm = Wilder_smoothing(minus_dm, period)
        smoothed_tr = Wilder_smoothing(tr, period)
        
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = Wilder_smoothing(dx, period)
        
        return adx, plus_di, minus_di
    
    adx_1d, _, _ = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    
    # Regime: trending when ADX > 25
    trending_regime = adx_aligned > 25
    
    # Calculate Camarilla levels on 1d data (based on previous day's OHLC)
    camarilla_r1_1d = close_1d + ((high_1d - low_1d) * 1.1 / 12)
    camarilla_s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d, additional_delay_bars=1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX and Camarilla
    start_idx = max(30, 20)  # 30 for ADX warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in trending regime with volume confirmation
            long_signal = (close[i] > camarilla_r1_aligned[i]) and trending_regime[i] and volume_spike[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and trending_regime[i] and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Camarilla S1 (mean reversion)
            exit_signal = close[i] < camarilla_s1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Camarilla R1 (mean reversion)
            exit_signal = close[i] > camarilla_r1_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dRegimeTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0