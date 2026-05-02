#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Uses 6h primary timeframe for signal generation with Elder Ray oscillator
# Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# 1d EMA50 trend filter provides higher timeframe bias (price > EMA50 for longs, < for shorts)
# Volume confirmation (1.5x 20-period average) filters for strong participation
# Discrete position sizing (0.25) balances profit potential with fee drag minimization
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Works in both bull and bear markets by only trading in direction of 1d trend
# Elder Ray measures bull/bear strength behind price moves, effective in ranging markets

name = "6h_ElderRay_1dEMA50_Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 and rising + volume spike + price > 1d EMA50
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and 
                volume_spike[i] and close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 and falling + volume spike + price < 1d EMA50
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and 
                  volume_spike[i] and close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power <= 0 or price < 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power >= 0 or price > 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals