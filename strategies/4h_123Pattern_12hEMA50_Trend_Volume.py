#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 123 Pattern + 12h EMA50 Trend + Volume Spike
# 123 Pattern: Price pulls back in a trend, creating a higher low (for uptrend) or lower high (for downtrend)
# Long when: 12h EMA50 uptrend + price makes higher low + breaks above prior swing high + volume spike
# Short when: 12h EMA50 downtrend + price makes lower high + breaks below prior swing low + volume spike
# This pattern captures trend continuation with low-risk entries, working in both bull and bear markets.
# Target: 20-50 trades/year to minimize fee drag.

name = "4h_123Pattern_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h swing points: identify swing highs and lows
    # Swing high: high[i] is highest in window of 5 bars (2 before, 2 after)
    # Swing low: low[i] is lowest in window of 5 bars
    window = 2
    swing_high = np.zeros(n, dtype=bool)
    swing_low = np.zeros(n, dtype=bool)
    
    for i in range(window, n - window):
        if high[i] == np.max(high[i-window:i+window+1]):
            swing_high[i] = True
        if low[i] == np.min(low[i-window:i+window+1]):
            swing_low[i] = True
    
    # Track most recent swing high and low
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    
    last_high_val = np.nan
    last_low_val = np.nan
    
    for i in range(n):
        if swing_high[i]:
            last_high_val = high[i]
        last_swing_high[i] = last_high_val
        
        if swing_low[i]:
            last_low_val = low[i]
        last_swing_low[i] = last_low_val
    
    # 12h EMA50 for trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(last_swing_high[i]) or 
            np.isnan(last_swing_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h EMA50 uptrend + higher low + breaks above prior swing high + volume spike
            # Higher low: current low > prior swing low
            # Breakout: current close > prior swing high
            ema_uptrend = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            higher_low = low[i] > last_swing_low[i]
            breakout_high = close[i] > last_swing_high[i]
            
            long_cond = ema_uptrend and higher_low and breakout_high and volume_spike[i]
            
            # Short: 12h EMA50 downtrend + lower high + breaks below prior swing low + volume spike
            # Lower high: current high < prior swing high
            # Breakdown: current close < prior swing low
            ema_downtrend = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
            lower_high = high[i] < last_swing_high[i]
            breakdown_low = close[i] < last_swing_low[i]
            
            short_cond = ema_downtrend and lower_high and breakdown_low and volume_spike[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below prior swing low (trend failure)
            if close[i] < last_swing_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above prior swing high (trend failure)
            if close[i] > last_swing_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals