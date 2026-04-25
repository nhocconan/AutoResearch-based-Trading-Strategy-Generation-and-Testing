#!/usr/bin/env python3
"""
6h_ADX_Williams_Alligator_Trend_Filter
Hypothesis: 6h trend following using Williams Alligator (SMAs with offsets) for trend direction and ADX(14) > 25 for trend strength confirmation. Uses 1d HTF for higher timeframe bias: only take longs when price > 1d EMA50, shorts when price < 1d EMA50. This combines multiple trend filters to reduce whipsaw in choppy markets while capturing strong trends. Designed to work in bull markets (strong uptrends with ADX>25) and bear markets (strong downtrends with ADX>25) by requiring confluence of Alligator alignment, ADX strength, and 1d EMA50 bias. Targets 12-25 trades/year per symbol by requiring strict trend alignment across multiple timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for HTF bias (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for HTF bias
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: three SMAs with offsets (Jaw=13, Teeth=8, Lips=5)
    # Alligator values are plotted forward: Jaw shifted by 8, Teeth by 5, Lips by 3
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8)
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw_vals = jaw.values
    teeth_vals = teeth.values
    lips_vals = lips.values
    
    # ADX(14) for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            high_diff = high[i] - high[i-1]
            low_diff = low[i-1] - low[i]
            
            plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
            minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing (alpha = 1/period)
        atr = np.zeros(len(high))
        atr[period] = np.mean(tr[1:period+1]) if len(tr) >= period+1 else 0
        
        plus_dm_smooth = np.zeros(len(high))
        minus_dm_smooth = np.zeros(len(high))
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (period-1) + minus_dm[i]) / period
        
        # Avoid division by zero
        dx = np.zeros(len(high))
        denom = plus_dm_smooth + minus_dm_smooth
        dx[denom != 0] = (abs(plus_dm_smooth[denom != 0] - minus_dm_smooth[denom != 0]) / denom[denom != 0]) * 100
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.mean(dx[period:2*period]) if len(dx) >= 2*period else 0
        
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_vals = calculate_adx(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Alligator (max shift 8) and ADX (2*14-1=27)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_vals[i]) or np.isnan(teeth_vals[i]) or np.isnan(lips_vals[i]) or
            np.isnan(adx_vals[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_vals[i] > teeth_vals[i] and teeth_vals[i] > jaw_vals[i]
        alligator_short = lips_vals[i] < teeth_vals[i] and teeth_vals[i] < jaw_vals[i]
        
        # ADX trend strength filter
        strong_trend = adx_vals[i] > 25
        
        # 1d HTF bias: price above/below 1d EMA50
        htf_bias_long = close[i] > ema50_1d_aligned[i]
        htf_bias_short = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Enter long: Alligator aligned up + strong trend + HTF bias long
            if alligator_long and strong_trend and htf_bias_long:
                signals[i] = 0.25
                position = 1
            # Enter short: Alligator aligned down + strong trend + HTF bias short
            elif alligator_short and strong_trend and htf_bias_short:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long position
            signals[i] = 0.25
            # Exit when Alligator loses alignment (Lips < Teeth) or ADX weakens
            exit_signal = lips_vals[i] < teeth_vals[i] or adx_vals[i] < 20
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short position
            signals[i] = -0.25
            # Exit when Alligator loses alignment (Lips > Teeth) or ADX weakens
            exit_signal = lips_vals[i] > teeth_vals[i] or adx_vals[i] < 20
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Williams_Alligator_Trend_Filter"
timeframe = "6h"
leverage = 1.0