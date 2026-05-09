# %%
#!/usr/bin/env python3
# Hypothesis: 6h Donchian breakout with weekly pivot direction and volume confirmation
# Long when price breaks above Donchian(20) high AND weekly pivot trend is up AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND weekly pivot trend is down AND volume > 1.5x 20-period average
# Exit when price crosses back below/above Donchian midpoint OR weekly trend contradicts position
# Designed to capture breakouts in trending markets with weekly pivot as trend filter and volume confirmation
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Weekly pivot trend determined by comparing current weekly close to previous weekly close

name = "6h_Donchian_WeeklyPivot_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donch_high = high_roll.values
    donch_low = low_roll.values
    donch_mid = (donch_high + donch_low) / 2
    
    # Get weekly data for pivot trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points: (H + L + C) / 3
    weekly_pivot = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    # Weekly trend: current week close > previous week close = uptrend
    weekly_trend = df_1w['close'] > df_1w['close'].shift(1)
    
    # Align weekly data to 6h timeframe (waits for weekly close)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot.values)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend.values.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian(20) and weekly data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_trend_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND weekly trend up AND volume spike
            if (close[i] > donch_high[i] and 
                weekly_trend_aligned[i] > 0.5 and  # True = uptrend
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low AND weekly trend down AND volume spike
            elif (close[i] < donch_low[i] and 
                  weekly_trend_aligned[i] < 0.5 and  # False = downtrend
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint OR weekly trend turns down
            if (close[i] < donch_mid[i]) or (weekly_trend_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint OR weekly trend turns up
            if (close[i] > donch_mid[i]) or (weekly_trend_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# %%