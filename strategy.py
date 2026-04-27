#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Williams %R with 1-week EMA trend filter and volume spike.
# Williams %R measures overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
# Strategy: In ranging markets, buy oversold (%R < -80) and sell overbought (%R > -20).
# In trending markets, follow 1-week EMA direction with pullback entries to %R extremes.
# Volume spike confirms institutional participation. Designed for ~15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    willr = -100 * ((highest_high - close) / hl_range)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(willr[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R levels
        oversold = -80
        overbought = -20
        
        # Oversold/overbought conditions
        is_oversold = willr[i] < oversold
        is_overbought = willr[i] > overbought
        
        if is_oversold and close[i] > ema34_1w_aligned[i] and volume_filter[i]:
            # Buy oversold in uptrend
            signals[i] = 0.25
            position = 1
        elif is_overbought and close[i] < ema34_1w_aligned[i] and volume_filter[i]:
            # Sell overbought in downtrend
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsR_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0