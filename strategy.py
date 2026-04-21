#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 Breakout with 12h EMA50 Trend Filter and Volume Spike Confirmation
# Long when price breaks above R1 in uptrend (12h EMA50 > EMA100), short when breaks below S1 in downtrend.
# Volume > 1.8x 30-period average confirms breakout. Uses EMA cross to filter weak trends and avoid chop.
# Camarilla levels provide strong intraday support/resistance; breakouts often trend.
# Target: 20-40 trades/year by requiring strong trend + volume + breakout alignment.
# Works in bull/bear: EMA filter ensures only trending markets are traded, avoiding whipsaws in ranges.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = close_prev + (high_prev - low_prev) * 1.1 / 12
    s1 = close_prev - (high_prev - low_prev) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 and EMA100 on 12h
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align EMA cross to 4h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema100_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)
    
    # Pre-compute volume moving average (30-period)
    vol_ma = prices['volume'].rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(ema100_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.8x 30-period average
        volume_confirm = volume > 1.8 * vol_ma[i]
        
        # Trend filter: uptrend if EMA50 > EMA100, downtrend if EMA50 < EMA100
        uptrend = ema50_aligned[i] > ema100_aligned[i]
        downtrend = ema50_aligned[i] < ema100_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 in uptrend
                if uptrend and price > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 in downtrend
                elif downtrend and price < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or trend reverses
                if price < s1_aligned[i] or ema50_aligned[i] < ema100_aligned[i]:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or trend reverses
                if price > r1_aligned[i] or ema50_aligned[i] > ema100_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0