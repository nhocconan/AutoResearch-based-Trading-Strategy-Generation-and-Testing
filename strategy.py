#!/usr/bin/env python3
# 1d_higher_low_higher_high_v1
# Hypothesis: 1d strategy using higher lows (for longs) and lower highs (for shorts) with 1w EMA200 trend filter and volume confirmation.
# Works in bull markets via higher low breakouts and in bear markets via lower high breakdowns.
# Discrete position sizing (±0.25) to minimize fee churn. Target: 30-100 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_higher_low_higher_high_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate higher lows and lower highs
    # Higher low: today's low > yesterday's low
    # Lower high: today's high < yesterday's high
    prev_low = np.roll(low, 1)
    prev_high = np.roll(high, 1)
    prev_low[0] = np.nan
    prev_high[0] = np.nan
    
    higher_low = low > prev_low
    lower_high = high < prev_high
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR higher low breaks
            if close[i] < ema200_1w_aligned[i] or not higher_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR lower high breaks
            if close[i] > ema200_1w_aligned[i] or not lower_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long: higher low with bullish trend
                if higher_low[i] and close[i] > ema200_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: lower high with bearish trend
                elif lower_high[i] and close[i] < ema200_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals