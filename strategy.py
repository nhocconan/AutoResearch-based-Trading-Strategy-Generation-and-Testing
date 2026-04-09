#!/usr/bin/env python3
# 6h_weekly_donchian_pullback_v1
# Hypothesis: 6h strategy using weekly Donchian channels for trend direction and daily pullbacks for entry.
# Long: Price above weekly Donchian(20) high and pulls back to touch 20-period EMA on 6h.
# Short: Price below weekly Donchian(20) low and pulls back to touch 20-period EMA on 6h.
# Exit: Price crosses the 20-period EMA in the opposite direction.
# Uses weekly structure for trend bias and 6h EMA for precise entry, reducing whipsaw in ranging markets.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_weekly_donchian_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 20-period EMA for pullback entries on 6h
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Get weekly data for Donchian channels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: rolling max of high
    high_1w_s = pd.Series(high_1w)
    donch_high = high_1w_s.rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    low_1w_s = pd.Series(low_1w)
    donch_low = low_1w_s.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to 6h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_20[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below EMA(20)
            if close[i] < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses above EMA(20)
            if close[i] > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for pullback entry in direction of weekly trend
            bullish_setup = (close[i] > donch_high_aligned[i]) and (close[i] <= ema_20[i])
            bearish_setup = (close[i] < donch_low_aligned[i]) and (close[i] >= ema_20[i])
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals