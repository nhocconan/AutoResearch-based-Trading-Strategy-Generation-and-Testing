#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d trend filter
# Uses 6h primary timeframe for optimal trade frequency (target: 12-37 trades/year)
# Williams Alligator (JAW/TEETH/LIPS) identifies trend structure and avoids chop
# Elder Ray (Bull Power/Bear Power) measures trend strength relative to EMA13
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend entries
# Designed for low trade frequency with 0.25 sizing to manage drawdown
# Works in bull markets via trend continuation and bear markets via trend-following alignment

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (HL/2) with specific periods
    # JAW: 13-period SMMA, TEETH: 8-period SMMA, LIPS: 5-period SMMA
    # Using EMA as approximation for SMMA (similar smoothing effect)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(median_price).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(median_price).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        # Williams Alligator signals: 
        # Bullish: Lips > Teeth > Jaw (all aligned upward)
        # Bearish: Lips < Teeth < Jaw (all aligned downward)
        alligator_bullish = lips[i] > teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] < jaw[i]
        
        # Elder Ray signals:
        # Bullish: Bull Power > 0 and increasing
        # Bearish: Bear Power < 0 and decreasing
        bull_power_rising = bull_power[i] > bull_power[i-1]
        bear_power_falling = bear_power[i] < bear_power[i-1]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and alligator_bullish and bull_power_rising and bull_power[i] > 0:
                # Long: bullish trend alignment with Elder Ray confirmation
                signals[i] = 0.25
                position = 1
            elif bearish_bias and alligator_bearish and bear_power_falling and bear_power[i] < 0:
                # Short: bearish trend alignment with Elder Ray confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: trend deterioration signals
            if (not alligator_bullish) or (bull_power[i] <= 0) or (not bullish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: trend deterioration signals
            if (not alligator_bearish) or (bear_power[i] >= 0) or (not bearish_bias):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals