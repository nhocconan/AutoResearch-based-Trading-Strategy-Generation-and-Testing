#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 6h Donchian(20) breakouts for entries
# Long when: 12h Supertrend is bullish AND price breaks above 6h Donchian upper(20) with volume > 1.5 * avg_volume(20)
# Short when: 12h Supertrend is bearish AND price breaks below 6h Donchian lower(20) with volume > 1.5 * avg_volume(20)
# Exit when: price crosses the 6h Donchian midpoint OR Supertrend flips direction
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Supertrend filters for 12h trend alignment, reducing false breakouts
# Donchian breakouts capture momentum in trending markets
# Volume confirmation ensures institutional participation
# Works in both bull (continuation breakouts in uptrend) and bear (continuation breakdowns in downtrend)

name = "6h_12hSupertrend_Donchian20_Breakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:  # Need sufficient data for ATR calculation
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(high_12h) - pd.Series(low_12h)
    tr2 = abs(pd.Series(high_12h) - pd.Series(close_12h).shift(1))
    tr3 = abs(pd.Series(low_12h) - pd.Series(close_12h).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_period, adjust=False, min_periods=atr_period).mean()
    
    # Basic Upperband and Lowerband
    basic_ub = (pd.Series(high_12h) + pd.Series(low_12h)) / 2.0 + multiplier * atr
    basic_lb = (pd.Series(high_12h) + pd.Series(low_12h)) / 2.0 - multiplier * atr
    
    # Final Upperband and Lowerband
    final_ub = pd.Series(index=range(len(high_12h)), dtype=float)
    final_lb = pd.Series(index=range(len(high_12h)), dtype=float)
    for i in range(len(high_12h)):
        if i == 0:
            final_ub.iloc[i] = basic_ub.iloc[i]
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            if basic_ub.iloc[i] < final_ub.iloc[i-1] or close_12h.iloc[i-1] > final_ub.iloc[i-1]:
                final_ub.iloc[i] = basic_ub.iloc[i]
            else:
                final_ub.iloc[i] = final_ub.iloc[i-1]
                
            if basic_lb.iloc[i] > final_lb.iloc[i-1] or close_12h.iloc[i-1] < final_lb.iloc[i-1]:
                final_lb.iloc[i] = basic_lb.iloc[i]
            else:
                final_lb.iloc[i] = final_lb.iloc[i-1]
    
    # Supertrend direction
    supertrend = pd.Series(index=range(len(high_12h)), dtype=float)
    direction = pd.Series(index=range(len(high_12h)), dtype=int)  # 1 for uptrend, -1 for downtrend
    for i in range(len(high_12h)):
        if i == 0:
            supertrend.iloc[i] = final_ub.iloc[i]
            direction.iloc[i] = 1
        else:
            if supertrend.iloc[i-1] == final_ub.iloc[i-1]:
                if close_12h.iloc[i] <= final_ub.iloc[i]:
                    supertrend.iloc[i] = final_ub.iloc[i]
                    direction.iloc[i] = 1
                else:
                    supertrend.iloc[i] = final_lb.iloc[i]
                    direction.iloc[i] = -1
            else:
                if close_12h.iloc[i] >= final_lb.iloc[i]:
                    supertrend.iloc[i] = final_lb.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = final_ub.iloc[i]
                    direction.iloc[i] = 1
    
    # Align 12h Supertrend direction to 6h timeframe (wait for completed 12h bar)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction.values.astype(float))
    
    # Calculate 6h Donchian channels (20-period)
    donchian_period = 20
    high_rolling = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max()
    low_rolling = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min()
    donchian_upper = high_rolling.values
    donchian_lower = low_rolling.values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(direction_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 12h Supertrend bullish AND price breaks above 6h Donchian upper with volume spike
            if (direction_aligned[i] == 1 and 
                close[i] > donchian_upper[i] and 
                close[i-1] <= donchian_upper[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: 12h Supertrend bearish AND price breaks below 6h Donchian lower with volume spike
            elif (direction_aligned[i] == -1 and 
                  close[i] < donchian_lower[i] and 
                  close[i-1] >= donchian_lower[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses 6h Donchian midpoint OR Supertrend flips to bearish
            if (close[i] <= donchian_mid[i]) or (direction_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses 6h Donchian midpoint OR Supertrend flips to bullish
            if (close[i] >= donchian_mid[i]) or (direction_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals