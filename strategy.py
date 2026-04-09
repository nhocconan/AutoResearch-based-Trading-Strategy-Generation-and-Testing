#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 12h EMA trend filter
# Williams %R(14) identifies oversold/overbought conditions: < -80 = oversold, > -20 = overbought
# 12h EMA(50) provides trend context: only long when price > EMA50, short when price < EMA50
# Entry: Williams %R crosses above -80 from below (oversold bounce) AND price > 12h EMA50 → long
# Entry: Williams %R crosses below -20 from above (overbought rejection) AND price < 12h EMA50 → short
# Exit: Opposite Williams %R cross (%R crosses below -50 for longs, above -50 for shorts) OR trend violation
# Works in bull/bear: mean reversion from extremes within trend context filters false signals
# Discrete position sizing: 0.0, ±0.25 to minimize fee churn
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_12h_williamsr_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50) for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Load 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)
    
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum fading) OR price < 12h EMA50 (trend violation)
            if williams_r_1d_aligned[i] < -50 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum fading) OR price > 12h EMA50 (trend violation)
            if williams_r_1d_aligned[i] > -50 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions with trend filter
            # Long: Williams %R crosses above -80 from below (oversold bounce) AND price > 12h EMA50
            # Short: Williams %R crosses below -20 from above (overbought rejection) AND price < 12h EMA50
            if i > 100:  # Need previous bar for crossover detection
                wr_prev = williams_r_1d_aligned[i-1]
                wr_curr = williams_r_1d_aligned[i]
                
                # Long entry: %R crosses above -80 from below
                long_entry = (wr_prev <= -80 and wr_curr > -80) and (close[i] > ema_50_12h_aligned[i])
                # Short entry: %R crosses below -20 from above
                short_entry = (wr_prev >= -20 and wr_curr < -20) and (close[i] < ema_50_12h_aligned[i])
                
                if long_entry:
                    position = 1
                    signals[i] = 0.25
                elif short_entry:
                    position = -1
                    signals[i] = -0.25
    
    return signals