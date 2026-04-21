#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla pivot levels with volume spike and 4h EMA21 trend filter.
Goes long when price closes above R1 with bullish trend and volume spike; goes short when price closes below S1 with bearish trend and volume spike.
Uses Camarilla levels from daily timeframe for institutional reference points, EMA21 for trend filter, and volume spike for confirmation.
Targets 20-40 trades/year (80-160 total over 4 years) to avoid fee drag while maintaining edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d Camarilla pivot levels (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data to calculate today's levels (no look-ahead)
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r1[i] = prev_close + (range_val * 1.1 / 12)
        camarilla_s1[i] = prev_close - (range_val * 1.1 / 12)
    
    # For first day, use zero (no levels available)
    camarilla_r1[0] = 0.0
    camarilla_s1[0] = 0.0
    
    # Align to 4h timeframe (wait for 1d bar to close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Load 4h data ONCE before loop for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21 = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        camarilla_r1 = camarilla_r1_aligned[i]
        camarilla_s1 = camarilla_s1_aligned[i]
        ema_trend = ema_21_aligned[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price closes above Camarilla R1 + uptrend + volume spike
            if (price_close > camarilla_r1 and 
                price_close > ema_trend and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below Camarilla S1 + downtrend + volume spike
            elif (price_close < camarilla_s1 and 
                  price_close < ema_trend and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses EMA21 in opposite direction)
            if position == 1 and price_close < ema_trend:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_CamarillaR1S1_4hEMA21_Volume"
timeframe = "4h"
leverage = 1.0