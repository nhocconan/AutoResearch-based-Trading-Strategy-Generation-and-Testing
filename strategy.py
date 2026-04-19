#!/usr/bin/env python3
"""
1d_WaveTrend_CCI_Arrows
Hypothesis: Daily WaveTrend (oscillator) with CCI filter for trend exhaustion.
WT1 > 60 and rising = bullish momentum, WT1 < -60 and falling = bearish momentum.
CCI(20) > 100 confirms overbought for short, < -100 confirms oversold for long.
Designed for 1d timeframe to target 30-100 total trades over 4 years (7-25/year).
Works in bull/bear via momentum exhaustion signals rather than pure trend following.
"""

name = "1d_WaveTrend_CCI_Arrows"
timeframe = "1d"
leverage = 1.0

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
    
    # WaveTrend calculation (similar to TCI)
    def wtc(high, low, close, channel_length=10, average_length=21):
        # Typical Price
        tp = (high + low + close) / 3.0
        
        # Exponential Moving Average of TP
        ema_tp = pd.Series(tp).ewm(span=channel_length, adjust=False).mean().values
        
        # Absolute deviation
        dev = np.abs(tp - ema_tp)
        
        # Smoothed deviation
        avg_dev = pd.Series(dev).ewm(span=channel_length, adjust=False).mean().values
        
        # Avoid division by zero
        ci = np.where(avg_dev != 0, (tp - ema_tp) / (0.015 * avg_dev), 0)
        
        # TCI (WaveTrend 1)
        tci = pd.Series(ci).ewm(span=average_length, adjust=False).mean().values
        
        # WaveTrend 2 (signal line)
        wt2 = pd.Series(tci).ewm(span=4, adjust=False).mean().values
        
        return tci, wt2
    
    # CCI calculation
    def cci(high, low, close, length=20):
        # Typical Price
        tp = (high + low + close) / 3.0
        
        # Simple Moving Average of TP
        sma_tp = pd.Series(tp).rolling(window=length, min_periods=length).mean().values
        
        # Mean Deviation
        md = np.zeros_like(tp)
        for i in range(length-1, len(tp)):
            md[i] = np.mean(np.abs(tp[i-length+1:i+1] - sma_tp[i]))
        
        # Avoid division by zero
        cci_val = np.where(md != 0, (tp - sma_tp) / (0.015 * md), 0)
        return cci_val
    
    # Calculate indicators on daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # WaveTrend
    wt1, wt2 = wtc(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 10, 21)
    wt1_aligned = align_htf_to_ltf(prices, df_1d, wt1)
    wt2_aligned = align_htf_to_ltf(prices, df_1d, wt2)
    
    # CCI
    cci_val = cci(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 20)
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci_val)
    
    # Volume confirmation: volume > 1.5 * 20-day average (reduced frequency)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wt1_aligned[i]) or np.isnan(wt2_aligned[i]) or 
            np.isnan(cci_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # WaveTrend conditions
        wt_bullish = wt1_aligned[i] > -60 and wt1_aligned[i] > wt2_aligned[i]  # WT1 above signal and above -60
        wt_bearish = wt1_aligned[i] < 60 and wt1_aligned[i] < wt2_aligned[i]   # WT1 below signal and below 60
        
        # CCI conditions for exhaustion
        cci_overbought = cci_aligned[i] > 100
        cci_oversold = cci_aligned[i] < -100
        
        if position == 0:
            # Long: WT bullish + CCI oversold (momentum turning up from oversold)
            if wt_bullish and cci_oversold and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: WT bearish + CCI overbought (momentum turning down from overbought)
            elif wt_bearish and cci_overbought and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit on WT bearish crossover or CCI overbought
            if (wt1_aligned[i] < wt2_aligned[i]) or cci_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit on WT bullish crossover or CCI oversold
            if (wt1_aligned[i] > wt2_aligned[i]) or cci_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals