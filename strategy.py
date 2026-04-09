#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams %R mean reversion with 1w trend filter
# - Uses 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure bull/bear strength
# - Enters long when Bear Power < 0 (bears weak) AND Williams %R < -80 (oversold)
# - Enters short when Bull Power > 0 (bulls strong) AND Williams %R > -20 (overbought)
# - Filters by 1w trend: only long when price > 1w EMA34, only short when price < 1w EMA34
# - Exits when Elder Ray power reverses (Bull Power < 0 for longs, Bear Power > 0 for shorts)
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-30 trades/year on 6h timeframe (48-120 total over 4 years) to minimize fee drag
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# - Elder Ray identifies power shifts, Williams %R provides timing, 1w EMA ensures trend alignment

name = "6h_1w_elderray_williamsr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_6h = get_htf_data(prices, '6h')
    if len(df_1w) < 40 or len(df_6h) < 40:
        return np.zeros(n)
    
    # Pre-compute 6h indicators
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # 6h EMA13 for Elder Ray
    ema13_6h = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 6h Elder Ray components
    bull_power = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_6h) / (highest_high_14 - lowest_low_14 + 1e-10)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(ema34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when Bear Power becomes positive (bears regain strength)
            if bear_power_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when Bull Power becomes negative (bulls lose strength)
            if bull_power_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries with trend filter
            # Long: Bear Power < 0 (bears weak) AND Williams %R < -80 (oversold) AND price > 1w EMA34 (uptrend)
            if (bear_power_aligned[i] < 0 and 
                williams_r_aligned[i] < -80 and 
                close[i] > ema34_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: Bull Power > 0 (bulls strong) AND Williams %R > -20 (overbought) AND price < 1w EMA34 (downtrend)
            elif (bull_power_aligned[i] > 0 and 
                  williams_r_aligned[i] > -20 and 
                  close[i] < ema34_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals