#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R + 1d Bollinger Band Mean Reversion
# - Williams %R (14) on 12h for short-term overbought/oversold signals
# - Bollinger Bands (20,2) on 1d to identify mean reversion zones
# - Long when Williams %R < -80 (oversold) and price < lower BB (undervalued)
# - Short when Williams %R > -20 (overbought) and price > upper BB (overvalued)
# - Combines momentum exhaustion with statistical extremes for high-conviction entries
# - Designed for 12h timeframe with selective entries to avoid overtrading
# - Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Bollinger Bands calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Bollinger Bands (20,2) on 1d timeframe
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean()
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std()
    upper_bb = (sma_20 + 2 * std_20).values
    lower_bb = (sma_20 - 2 * std_20).values
    
    # Align 1d Bollinger Bands to 12h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    
    # Calculate Williams %R (14) on 12h timeframe
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    highest_high_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_12h) / (highest_high_14 - lowest_low_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after Williams %R warmup
        # Skip if NaN in indicators
        if np.isnan(williams_r[i]) or np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        wr = williams_r[i]
        upper = upper_bb_aligned[i]
        lower = lower_bb_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) and price below lower BB (undervalued)
            if wr < -80 and price < lower:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) and price above upper BB (overvalued)
            elif wr > -20 and price > upper:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or price reaches middle of BB
            middle_bb = (upper + lower) / 2
            if wr > -50 or price > middle_bb:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or price reaches middle of BB
            middle_bb = (upper + lower) / 2
            if wr < -50 or price < middle_bb:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dBB_MeanReversion"
timeframe = "12h"
leverage = 1.0