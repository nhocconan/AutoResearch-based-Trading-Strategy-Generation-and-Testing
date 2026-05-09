#!/usr/bin/env python3
# Hypothesis: 12h timeframe with 1-day Williams %R extreme readings and 12-hour RSI momentum confirmation.
# In oversold conditions (Williams %R < -80), price tends to revert upward; in overbought (Williams %R > -20), price tends to revert downward.
# Uses 12-hour RSI to confirm momentum: only take longs when RSI > 50 (bullish momentum), shorts when RSI < 50 (bearish momentum).
# Williams %R calculated on daily timeframe provides institutional-level overbought/oversold signals.
# RSI on 12h provides entry timing aligned with the primary timeframe.
# Exit when Williams %R returns to neutral range (-50 to -50) or RSI crosses 50 in opposite direction.
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25.

name = "12h_WilliamsR_RSI_Momentum"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 1-day Williams %R (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = high_1d.rolling(window=14, min_periods=14).max()
    lowest_low = low_1d.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    
    williams_r_values = williams_r.values
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_values)
    
    # Oversold/overbought conditions
    williams_r_oversold = williams_r < -80
    williams_r_overbought = williams_r > -20
    williams_r_oversold_values = williams_r_oversold.values
    williams_r_overbought_values = williams_r_overbought.values
    williams_r_oversold_aligned = align_htf_to_ltf(prices, df_1d, williams_r_oversold_values)
    williams_r_overbought_aligned = align_htf_to_ltf(prices, df_1d, williams_r_overbought_values)
    
    # 12-hour RSI (14-period) for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    rsi_values = rsi.values
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50
    rsi_above_50_values = rsi_above_50.values
    rsi_below_50_values = rsi_below_50.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r_aligned[i]) or
            np.isnan(williams_r_oversold_aligned[i]) or np.isnan(williams_r_overbought_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(rsi_above_50_values[i]) or np.isnan(rsi_below_50_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Williams %R oversold + RSI > 50 (bullish momentum)
            if williams_r_oversold_aligned[i] and rsi_above_50_values[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R overbought + RSI < 50 (bearish momentum)
            elif williams_r_overbought_aligned[i] and rsi_below_50_values[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R exits oversold OR RSI turns bearish (< 50)
            if (not williams_r_oversold_aligned[i]) or (not rsi_above_50_values[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R exits overbought OR RSI turns bullish (> 50)
            if (not williams_r_overbought_aligned[i]) or (not rsi_below_50_values[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals