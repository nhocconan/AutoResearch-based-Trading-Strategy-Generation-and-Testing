#!/usr/bin/env python3
"""
12H Bollinger Band Squeeze Breakout with Volume Confirmation
Hypothesis: Bollinger Band width at 30-day low indicates low volatility compression.
Breakout above upper band or below lower band with volume >1.5x average captures explosive moves.
Works in both bull (bullish breakouts) and bear (bearish breakouts) markets.
Target: 15-30 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_bb_squeeze_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2) on 12h
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    bb_width = upper - lower
    
    # Bollinger Band width percentile (30-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=30, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below middle band (mean reversion)
            if close[i] < sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle band (mean reversion)
            if close[i] > sma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Bollinger squeeze condition: BB width at 20th percentile or lower (low volatility)
            squeeze_condition = bb_width_percentile[i] <= 20
            
            # Breakout above upper band with volume
            if (close[i] >= upper[i] and 
                squeeze_condition and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout below lower band with volume
            elif (close[i] <= lower[i] and 
                  squeeze_condition and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals