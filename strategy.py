#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA50 trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13. Trades in direction of 1d trend.
# Works in bull/bear markets by requiring alignment with higher timeframe trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
name = "6h_ElderRay_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (6h timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for EMA13 and volume calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_6h[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(volume_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        bullish_signal = bull_power[i] > 0  # Positive bull power
        bearish_signal = bear_power[i] < 0  # Negative bear power
        trend_up = close[i] > ema_50_6h[i]
        trend_down = close[i] < ema_50_6h[i]
        
        if position == 0:
            # Long: bullish power + uptrend + volume confirmation
            if bullish_signal and trend_up and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish power + downtrend + volume confirmation
            elif bearish_signal and trend_down and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: loss of bullish power or trend reversal
            if bull_power[i] <= 0 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: loss of bearish power or trend reversal
            if bear_power[i] >= 0 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals