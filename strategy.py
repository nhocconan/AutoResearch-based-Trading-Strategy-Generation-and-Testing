# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform with Weekly Trend Filter
# - Uses Ehlers Fisher Transform (9-period) on 6h timeframe for mean-reversion signals
# - Long when Fisher crosses above -1.5, short when crosses below +1.5
# - Weekly trend filter ensures we only trade in direction of weekly trend
# - Works in bull/bear by using weekly trend to avoid counter-trend trades
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 6h timeframe

name = "6h_FisherTransform_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 6h data for Fisher Transform
    if n < 50:
        return np.zeros(n)
    
    # Ehlers Fisher Transform (9-period)
    # Step 1: Normalize price to [-1, 1] range over period
    period = 9
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest_high - lowest_low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    
    # Normalize price to [0, 1] then to [-1, 1]
    value1 = 2 * ((close - lowest_low) / range_val) - 1
    value1 = np.clip(value1, -0.999, 0.999)  # Prevent log(0)
    
    # Step 2: Apply Fisher Transform
    fish = np.zeros(n)
    fish[0] = 0
    for i in range(1, n):
        fish[i] = 0.5 * np.log((1 + value1[i]) / (1 - value1[i])) + 0.5 * fish[i-1]
    
    # Step 3: Apply smoothing
    fish_smoothed = np.zeros(n)
    fish_smoothed[0] = fish[0]
    for i in range(1, n):
        fish_smoothed[i] = 0.5 * fish[i] + 0.5 * fish_smoothed[i-1]
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for Fisher and weekly alignment
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(fish_smoothed[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Fisher crosses above -1.5 with weekly uptrend
            long_cross = fish_smoothed[i] > -1.5 and fish_smoothed[i-1] <= -1.5
            weekly_up = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            
            # Short: Fisher crosses below +1.5 with weekly downtrend
            short_cross = fish_smoothed[i] < 1.5 and fish_smoothed[i-1] >= 1.5
            weekly_down = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
            
            if long_cross and weekly_up:
                signals[i] = 0.25
                position = 1
            elif short_cross and weekly_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Fisher crosses below +1.5 (mean reversion complete)
            if fish_smoothed[i] < 1.5 and fish_smoothed[i-1] >= 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Fisher crosses above -1.5 (mean reversion complete)
            if fish_smoothed[i] > -1.5 and fish_smoothed[i-1] <= -1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%