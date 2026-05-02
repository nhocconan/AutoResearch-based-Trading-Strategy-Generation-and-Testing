#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA50 trend filter + volume confirmation
# Targets 12-37 trades per year (50-150 total over 4 years) on 6h timeframe
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via smoothed medians
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms institutional participation
# Works in both bull and bear: trend filter prevents counter-trend, Alligator catches reversals
# Discrete position sizing 0.25 minimizes fee drag while maintaining exposure

name = "6h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h data
    # Alligator uses SMMA (Smoothed Moving Average) of MEDIAN price (HL/2)
    median_price = (high + low) / 2.0
    
    # Smoothed Moving Average (SMMA) calculation
    def smma(source, period):
        sma = np.zeros_like(source)
        sma[:] = np.nan
        if len(source) >= period:
            sma[period-1] = np.mean(source[:period])
            for i in range(period, len(source)):
                sma[i] = (sma[i-1] * (period-1) + source[i]) / period
        return sma
    
    jaw = smma(median_price, 13)  # Blue line
    teeth = smma(median_price, 8)  # Red line
    lips = smma(median_price, 5)   # Green line
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator: max period 13 + buffer)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
            bullish_alligator = lips[i] > teeth[i] > jaw[i]
            bearish_alligator = lips[i] < teeth[i] < jaw[i]
            
            # Long: Bullish Alligator AND price > 1d EMA50 AND volume spike
            if (bullish_alligator and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND price < 1d EMA50 AND volume spike
            elif (bearish_alligator and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment OR price < 1d EMA50
            bearish_exit = lips[i] < teeth[i] < jaw[i]
            if bearish_exit or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment OR price > 1d EMA50
            bullish_exit = lips[i] > teeth[i] > jaw[i]
            if bullish_exit or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals