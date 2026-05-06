#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d trend filter and volume confirmation
# Long: Bull Power > 0 (price > EMA13) AND Bear Power < 0 (strong bears weakening) AND price > 1d EMA50 (uptrend) AND volume > 1.5 * 20-bar avg
# Short: Bear Power < 0 (price < EMA13) AND Bull Power < 0 (weak bulls) AND price < 1d EMA50 (downtrend) AND volume > 1.5 * 20-bar avg
# Exit when Bull Power and Bear Power converge (both near zero) indicating weakening momentum
# Uses discrete sizing 0.25 to balance opportunity and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear power relative to EMA13; convergence signals exhaustion
# 1d EMA50 provides higher-timeframe trend alignment; volume confirms participation

name = "6h_ElderRay_1dEMA50_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: high - EMA13
    bear_power = low - ema13   # Bear Power: low - EMA13
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions
            # Long: Bull Power > 0 AND Bear Power < 0 (bulls in control, bears weakening) 
            #       AND uptrend AND volume spike
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close[i] > ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power < 0 (bears in control, bulls weak)
            #        AND downtrend AND volume spike
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  close[i] < ema_50_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power weakening (< 0.5) OR Bear Power strengthening (> -0.5)
            # Indicates momentum convergence
            if bull_power[i] < 0.5 or bear_power[i] > -0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power weakening (> -0.5) OR Bull Power strengthening (< 0.5)
            if bear_power[i] > -0.5 or bull_power[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals