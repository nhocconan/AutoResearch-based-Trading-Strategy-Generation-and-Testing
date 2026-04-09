#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h Elder Ray (Bull/Bear Power) confluence
# - Williams %R(14) on 6h for overbought/oversold conditions
# - Elder Ray on 12h: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# - Long: 6h Williams %R < -80 (oversold) AND 12h Bull Power > 0 AND rising
# - Short: 6h Williams %R > -20 (overbought) AND 12h Bear Power > 0 AND rising
# - Uses EMA(13) for Elder Ray calculation with proper alignment
# - Position size: 0.25 to manage drawdown in volatile markets
# - Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend)
# - Elder Ray confirms trend direction from higher timeframe
# - Target: 12-25 trades/year on 6h (50-100 total over 4 years) to minimize fee drag

name = "6h_12h_williams_elderray_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators for Elder Ray
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA(13) for Elder Ray
    close_12h_series = pd.Series(close_12h)
    ema_13_12h = close_12h_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 12h Elder Ray components
    bull_power_12h = high_12h - ema_13_12h  # Bull Power = High - EMA(13)
    bear_power_12h = ema_13_12h - low_12h   # Bear Power = EMA(13) - Low
    
    # Align 12h Elder Ray to 6h
    bull_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_12h_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 6h Williams %R(14)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_12h_aligned[i]) or 
            np.isnan(bear_power_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R rises above -50 (exit oversold) OR Bear Power becomes strong
            if williams_r[i] > -50 or bear_power_12h_aligned[i] > 2.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R falls below -50 (exit overbought) OR Bull Power becomes strong
            if williams_r[i] < -50 or bull_power_12h_aligned[i] > 2.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for entry conditions
            # Long: Williams %R oversold (< -80) AND Bull Power positive AND rising
            # Short: Williams %R overbought (> -20) AND Bear Power positive AND rising
            
            # Check if we have enough history for rising/falling check
            if i >= 1:
                bull_power_rising = bull_power_12h_aligned[i] > bull_power_12h_aligned[i-1]
                bear_power_rising = bear_power_12h_aligned[i] > bear_power_12h_aligned[i-1]
                
                # Long condition
                if (williams_r[i] < -80 and 
                    bull_power_12h_aligned[i] > 0 and 
                    bull_power_rising):
                    position = 1
                    signals[i] = 0.25
                # Short condition
                elif (williams_r[i] > -20 and 
                      bear_power_12h_aligned[i] > 0 and 
                      bear_power_rising):
                    position = -1
                    signals[i] = -0.25
    
    return signals