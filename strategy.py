#!/usr/bin/env python3
# 1d_donchian_breakout_1w_trend_v1
# Hypothesis: Use weekly Donchian channel breakout with daily trend filter.
# Long when price breaks above 1-week Donchian high with bullish daily trend (price > 50-day EMA).
# Short when price breaks below 1-week Donchian low with bearish daily trend (price < 50-day EMA).
# Exit when price crosses opposite Donchian boundary or trend reverses.
# Uses weekly structure to capture major trends while reducing trade frequency.
# Target: 15-25 trades/year to minimize fee decay while capturing trend moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-week period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 20-period rolling max/min for Donchian bands
    high_roll = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_roll)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_roll)
    
    # Daily trend filter: 50-period EMA
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    uptrend = close > ema50
    downtrend = close < ema50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema50[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly Donchian low or trend turns bearish
            if close[i] < donchian_low[i] or downtrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly Donchian high or trend turns bullish
            if close[i] > donchian_high[i] or uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above weekly Donchian high with bullish trend
            if close[i] > donchian_high[i] and uptrend[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below weekly Donchian low with bearish trend
            elif close[i] < donchian_low[i] and downtrend[i]:
                position = -1
                signals[i] = -0.25
    
    return signals