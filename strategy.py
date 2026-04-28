#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for overbought/oversold conditions and 6h EMA crossover for entry.
# Williams %R identifies extreme levels on higher timeframe (1d) to avoid chop. EMA(9,21) crossover captures momentum in direction of extreme reversion.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) regimes.
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "6h_WilliamsR1d_EMACrossover_Extreme_v1"
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
    
    # Get 1d data for Williams %R (extreme filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_1d[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    
    # Align Williams %R to 6h
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 6h EMA(9) and EMA(21) for crossover
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(ema9[i]) or
            np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Extreme conditions from 1d Williams %R
        oversold = williams_r_1d_aligned[i] <= -80  # extreme oversold
        overbought = williams_r_1d_aligned[i] >= -20  # extreme overbought
        
        # EMA crossover conditions
        ema_bullish = ema9[i] > ema21[i]  # bullish momentum
        ema_bearish = ema9[i] < ema21[i]  # bearish momentum
        
        # Entry conditions: buy oversold dips in bullish momentum, sell overbought rallies in bearish momentum
        long_entry = oversold and ema_bullish
        short_entry = overbought and ema_bearish
        
        # Exit: opposite EMA crossover
        long_exit = ema_bearish  # exit long when momentum turns bearish
        short_exit = ema_bullish  # exit short when momentum turns bullish
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R for overbought/oversold conditions and 6h EMA crossover for entry.
# Williams %R identifies extreme levels on higher timeframe (1d) to avoid chop. EMA(9,21) crossover captures momentum in direction of extreme reversion.
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend) regimes.
# Target: 50-150 trades over 4 years (12-37/year). Size: 0.25.

name = "6h_WilliamsR1d_EMACrossover_Extreme_v1"
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
    
    # Get 1d data for Williams %R (extreme filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_1d[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    
    # Align Williams %R to 6h
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # 6h EMA(9) and EMA(21) for crossover
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(ema9[i]) or
            np.isnan(ema21[i])):
            signals[i] = 0.0
            continue
        
        # Extreme conditions from 1d Williams %R
        oversold = williams_r_1d_aligned[i] <= -80  # extreme oversold
        overbought = williams_r_1d_aligned[i] >= -20  # extreme overbought
        
        # EMA crossover conditions
        ema_bullish = ema9[i] > ema21[i]  # bullish momentum
        ema_bearish = ema9[i] < ema21[i]  # bearish momentum
        
        # Entry conditions: buy oversold dips in bullish momentum, sell overbought rallies in bearish momentum
        long_entry = oversold and ema_bullish
        short_entry = overbought and ema_bearish
        
        # Exit: opposite EMA crossover
        long_exit = ema_bearish  # exit long when momentum turns bearish
        short_exit = ema_bullish  # exit short when momentum turns bullish
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_WilliamsR1d_EMACrossover_Extreme_v1"
timeframe = "6h"
leverage = 1.0