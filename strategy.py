#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for monthly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate monthly pivots (using previous month's high/low/close)
    # We'll use a 20-day lookback for monthly high/low/close approximation
    high_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    close_20d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Monthly pivot points
    monthly_pivot = (high_20d + low_20d + close_20d) / 3.0
    monthly_r1 = 2 * monthly_pivot - low_20d
    monthly_s1 = 2 * monthly_pivot - high_20d
    monthly_r2 = monthly_pivot + (high_20d - low_20d)
    monthly_s2 = monthly_pivot - (high_20d - low_20d)
    
    # Shift to use previous month's data (avoid look-ahead)
    monthly_r1_prev = np.roll(monthly_r1, 1)
    monthly_s1_prev = np.roll(monthly_s1, 1)
    monthly_r2_prev = np.roll(monthly_r2, 1)
    monthly_s2_prev = np.roll(monthly_s2, 1)
    monthly_r1_prev[0] = np.nan
    monthly_s1_prev[0] = np.nan
    monthly_r2_prev[0] = np.nan
    monthly_s2_prev[0] = np.nan
    
    # Align monthly pivots to 6h timeframe
    monthly_r1_6h = align_htf_to_ltf(prices, df_1d, monthly_r1_prev)
    monthly_s1_6h = align_htf_to_ltf(prices, df_1d, monthly_s1_prev)
    monthly_r2_6h = align_htf_to_ltf(prices, df_1d, monthly_r2_prev)
    monthly_s2_6h = align_htf_to_ltf(prices, df_1d, monthly_s2_prev)
    
    # Volume confirmation: current volume > 2.0 * 6-period average (6h * 6 = 36h)
    volume_ma6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # ATR filter to avoid low volatility environments
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma10 = pd.Series(atr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need 20-day lookback and ATR MA10
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(volume_ma6[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(atr_ma10[i]) or 
            np.isnan(monthly_r1_6h[i]) or 
            np.isnan(monthly_s1_6h[i]) or
            np.isnan(monthly_r2_6h[i]) or 
            np.isnan(monthly_s2_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 6-period average
        volume_filter = volume[i] > (2.0 * volume_ma6[i])
        # Volatility filter: ATR > ATR MA10 (avoid low volatility)
        volatility_filter = atr[i] > atr_ma10[i]
        
        if position == 0:
            # Long: price breaks above monthly R2 with volume and volatility (strong breakout)
            if close[i] > monthly_r2_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below monthly S2 with volume and volatility (strong breakdown)
            elif close[i] < monthly_s2_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
            # Long reversal: price rejects monthly S1 and moves back above it (bullish rejection)
            elif close[i] > monthly_s1_6h[i] and low[i] < monthly_s1_6h[i] and volume_filter and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short reversal: price rejects monthly R1 and moves back below it (bearish rejection)
            elif close[i] < monthly_r1_6h[i] and high[i] > monthly_r1_6h[i] and volume_filter and volatility_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below monthly R1 or volatility drops
            if close[i] < monthly_r1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above monthly S1 or volatility drops
            if close[i] > monthly_s1_6h[i] or not volatility_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_MonthlyPivot_R2_S2_Breakout_Rejection_Vol"
timeframe = "6h"
leverage = 1.0