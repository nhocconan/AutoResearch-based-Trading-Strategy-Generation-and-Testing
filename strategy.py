#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA34 Trend + Volume Confirmation
Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (all lines intertwined) vs presence (diverged lines).
In trending markets (price above/below all Alligator lines), we enter long/short with 1d EMA34 trend filter and volume spike.
Uses discrete sizing 0.25 to limit fee churn. Timeframe 12h targets 12-37 trades/year.
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
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3) - all SMMA
    def smma(source, period):
        result = np.full_like(source, np.nan, dtype=np.float64)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price_12h, 13)
    teeth = smma(median_price_12h, 8)
    lips = smma(median_price_12h, 5)
    
    # Align Alligator lines to 12h timeframe (primary)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 13)  # need EMA34_1d, vol MA, Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above all Alligator lines (bullish alignment) AND price > 1d EMA34 (uptrend) AND volume spike
            if (close[i] > jaw_aligned[i] and close[i] > teeth_aligned[i] and close[i] > lips_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below all Alligator lines (bearish alignment) AND price < 1d EMA34 (downtrend) AND volume spike
            elif (close[i] < jaw_aligned[i] and close[i] < teeth_aligned[i] and close[i] < lips_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price re-enters Alligator's mouth (lines intertwine) OR loss of 1d EMA34 trend
            exit_signal = False
            if position == 1:
                # Exit long when price closes below Jaw (trend weakness) OR price < 1d EMA34
                if close[i] < jaw_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when price closes above Jaw (trend weakness) OR price > 1d EMA34
                if close[i] > jaw_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0