#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Spike + Weekly Trend Filter
# Long when price > Alligator Jaw (TEMA13) and 1d volume > 2x 20-period average and weekly close > weekly EMA34
# Short when price < Alligator Jaw (TEMA13) and 1d volume > 2x 20-period average and weekly close < weekly EMA34
# Exit when price crosses Alligator Teeth (TEMA8)
# Williams Alligator uses smoothed moving averages (SMMA) to identify trends
# Volume confirms breakout strength
# Weekly trend filter ensures alignment with higher timeframe trend
# Target: 15-30 trades/year by requiring weekly trend + volume spike + Alligator alignment

def tema(arr, period):
    """Triple Exponential Moving Average"""
    ema1 = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean()
    ema2 = ema1.ewm(span=period, adjust=False, min_periods=period).mean()
    ema3 = ema2.ewm(span=period, adjust=False, min_periods=period).mean()
    return 3 * (ema1 - ema2) + ema3

def smma(arr, period):
    """Smoothed Moving Average (Wilder's smoothing)"""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: smoothed
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Williams Alligator (using SMMA)
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)  # TEMA13
    teeth = smma(close_1d, 8)  # TEMA8
    lips = smma(close_1d, 5)   # TEMA5
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d volume moving average (20-period)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Wait for weekly EMA34
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price
        price = prices['close'].iloc[i]
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        vol_ma = vol_ma_1d_aligned[i]
        volume_confirm = df_1d['volume'].iloc[i] > 2 * vol_ma
        
        # Weekly trend filter
        weekly_uptrend = close_1w.iloc[i] > ema34_1w.iloc[i] if i < len(close_1w) else close_1w.iloc[-1] > ema34_1w.iloc[-1]
        weekly_downtrend = close_1w.iloc[i] < ema34_1w.iloc[i] if i < len(close_1w) else close_1w.iloc[-1] < ema34_1w.iloc[-1]
        
        if position == 0:
            if volume_confirm:
                # Long: price > Jaw and weekly uptrend
                if price > jaw_aligned[i] and weekly_uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: price < Jaw and weekly downtrend
                elif price < jaw_aligned[i] and weekly_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit when price crosses Teeth
            exit_signal = False
            
            if position == 1:  # long position
                if price < teeth_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                if price > teeth_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dVolumeSpike_1wEMA34_Trend"
timeframe = "12h"
leverage = 1.0