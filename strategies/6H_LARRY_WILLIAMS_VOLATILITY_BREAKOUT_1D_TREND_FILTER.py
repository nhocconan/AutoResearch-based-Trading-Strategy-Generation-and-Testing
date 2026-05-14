#!/usr/bin/env python3
# 6H_LARRY_WILLIAMS_VOLATILITY_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Larry Williams Volatility Breakout captures momentum breakouts from volatility contractions.
# Combines with 1D trend filter to avoid counter-trend trades. Works in both bull and bear markets:
# - In bull markets: captures breakout continuations
# - In bear markets: captures sharp reversals after volatility spikes
# Target: 15-25 trades/year on 6h timeframe.

name = "6H_LARRY_WILLIAMS_VOLATILITY_BREAKOUT_1D_TREND_FILTER"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily data for trend filter and volatility calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily range for volatility (ATR-like)
    daily_range = df_1d['high'].values - df_1d['low'].values
    # Average True Range approximation using daily ranges
    atr_approx = pd.Series(daily_range).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # EMA34 for trend filter
    ema34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 6h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_approx)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(atr_aligned[i]) or np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Larry Williams Volatility Breakout calculation
        # Long breakout: open + k * previous day's range
        # Short breakout: open - k * previous day's range
        k = 0.5  # Volatility multiplier
        
        if i > 0:
            # Get previous day's range from aligned ATR
            prev_range = atr_aligned[i-1]
            long_trigger = prices['open'].values[i] + k * prev_range
            short_trigger = prices['open'].values[i] - k * prev_range
            
            if position == 0:
                # LONG: Price breaks above long trigger in uptrend
                if (close[i] > long_trigger and 
                    close[i] > ema34_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                # SHORT: Price breaks below short trigger in downtrend
                elif (close[i] < short_trigger and 
                      close[i] < ema34_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif position == 1:
                # EXIT LONG: Price falls below short trigger or trend reversal
                if (close[i] < short_trigger or 
                    close[i] <= ema34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # EXIT SHORT: Price rises above long trigger or trend reversal
                if (close[i] > long_trigger or 
                    close[i] >= ema34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals