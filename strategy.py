#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray combination with 1w trend filter.
# Uses 1w EMA50 as trend filter and Williams Alligator (jaw/teeth/lips) for entry signals.
# Elder Ray (bull/bear power) confirms momentum direction.
# Works in bull (buy when teeth > lips and bull power > 0 with uptrend) and bear 
# (sell when teeth < lips and bear power < 0 with downtrend).
# Discrete position sizing 0.25 balances return and drawdown. Target: 30-100 trades over 4 years.

name = "1d_WilliamsAlligator_ElderRay_1wTrend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: SMAs of median price with different periods
    # Median price = (high + low) / 2
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # Using SMA as approximation for SMMA (simplified)
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray Index: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(13, 8, 5) + 8  # 21 (accounts for Alligator shifts)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 1w EMA50 direction
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Williams Alligator conditions
        # Teeth > Lips = bullish alignment
        # Teeth < Lips = bearish alignment
        bullish_alligator = teeth[i] > lips[i]
        bearish_alligator = teeth[i] < lips[i]
        
        # Elder Ray conditions
        bull_power_positive = bull_power[i] > 0
        bear_power_negative = bear_power[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator AND Bull Power positive AND uptrend
            if bullish_alligator and bull_power_positive and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator AND Bear Power negative AND downtrend
            elif bearish_alligator and bear_power_negative and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on bearish Alligator alignment (teeth < lips)
            if teeth[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on bullish Alligator alignment (teeth > lips)
            if teeth[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals