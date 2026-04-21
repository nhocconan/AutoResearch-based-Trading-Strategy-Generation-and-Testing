#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA200 Trend Filter + Volume Spike
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (13-period EMA on 6h)
# Long when Bull Power > 0 and Bear Power turning up (from negative) in uptrend (price > 1d EMA200)
# Short when Bear Power < 0 and Bull Power turning down (from positive) in downtrend (price < 1d EMA200)
# Volume spike (>1.5x 20-period average) confirms conviction
# Works in bull/bear: 1d EMA200 filter ensures we trade with higher timeframe trend
# Target: 20-35 trades/year by requiring EMA200 trend + Elder Ray divergence + volume

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 13-period EMA for Elder Ray (on 6d data)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = low - ema13   # Low - EMA13
    
    # Previous values for crossover detection
    bull_power_prev = np.roll(bull_power, 1)
    bear_power_prev = np.roll(bear_power, 1)
    bull_power_prev[0] = np.nan
    bear_power_prev[0] = np.nan
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price vs 1d EMA200
        uptrend = price > ema200_1d_aligned[i]
        downtrend = price < ema200_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: Bull Power > 0 AND Bear Power turning up (from negative) in uptrend
                if bull_power[i] > 0 and bear_power[i] > bear_power_prev[i] and bear_power_prev[i] <= 0 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: Bear Power < 0 AND Bull Power turning down (from positive) in downtrend
                elif bear_power[i] < 0 and bull_power[i] < bull_power_prev[i] and bull_power_prev[i] >= 0 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if Bull Power turns negative or trend fails
                if bull_power[i] <= 0 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if Bear Power turns positive or trend fails
                if bear_power[i] >= 0 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_1dEMA200_Trend_Volume"
timeframe = "6h"
leverage = 1.0