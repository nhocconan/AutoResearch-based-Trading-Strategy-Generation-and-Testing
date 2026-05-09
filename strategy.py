#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Alligator_ElderRay_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and Alligator
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Alligator Jaw
    close_1d = pd.Series(df_1d['close'].values)
    ema13 = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # Calculate 1d EMA8 for Alligator Teeth
    ema8 = close_1d.ewm(span=8, adjust=False, min_periods=8).mean().values
    ema8_aligned = align_htf_to_ltf(prices, df_1d, ema8)
    
    # Calculate 1d EMA5 for Alligator Lips
    ema5 = close_1d.ewm(span=5, adjust=False, min_periods=5).mean().values
    ema5_aligned = align_htf_to_ltf(prices, df_1d, ema5)
    
    # Calculate 13-period EMA for Elder Ray Bull/Bear Power
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = (df_1d['high'].values - ema13_1d)  # High - EMA13
    bear_power = (df_1d['low'].values - ema13_1d)   # Low - EMA13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # 6-period RSI for entry timing
    rsi_period = 6
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(ema8_aligned[i]) or np.isnan(ema5_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions: aligned for trend
        jaw = ema13_aligned[i]
        teeth = ema8_aligned[i]
        lips = ema5_aligned[i]
        
        # Elder Ray: positive bull power = buying pressure, negative bear power = selling pressure
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        
        if position == 0:
            # Long: Alligator aligned up (lips > teeth > jaw) + bull power positive + RSI not overbought
            if lips > teeth > jaw and bull > 0 and rsi[i] < 70:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down (lips < teeth < jaw) + bear power negative + RSI not oversold
            elif lips < teeth < jaw and bear < 0 and rsi[i] > 30:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns down OR bear power turns negative
            if lips < teeth or bear < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns up OR bull power turns positive
            if lips > teeth or bull > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals