#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d Supertrend + Volume Spike
# - Williams %R(14): Overbought > -20, Oversold < -80
# - 1d Supertrend(ATR=10, mult=3.0): Determines primary trend direction
# - Long when Williams %R crosses above -80 from below AND 1d Supertrend = uptrend AND volume > 2.0x 20-period average
# - Short when Williams %R crosses below -20 from above AND 1d Supertrend = downtrend AND volume > 2.0x 20-period average
# - Exit when Williams %R crosses opposite threshold (-20 for long, -80 for short) or Supertrend flips
# - Volume confirmation prevents false signals in low-participation moves
# - Williams %R excels at catching reversals in both bull and bear markets
# - Supertrend filter ensures we only trade with the higher timeframe trend
# - Targets ~20-30 trades/year (80-120 total over 4 years) to balance opportunity and fee drag

name = "6h_1d_williamsr_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0,
                          ((highest_high - close) / (highest_high - lowest_low)) * -100,
                          -50.0)  # Neutral when range is zero
    
    # Pre-compute 1d Supertrend components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1d + low_1d) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Initialize Supertrend arrays
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    # Set first valid value
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
        
        # Ensure Supertrend stays within bounds
        if direction[i] == 1 and supertrend[i] > upper_band[i]:
            direction[i] = -1
            supertrend[i] = upper_band[i]
        elif direction[i] == -1 and supertrend[i] < lower_band[i]:
            direction[i] = 1
            supertrend[i] = lower_band[i]
    
    # Align 1d Supertrend direction to 6h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Pre-compute volume confirmation: > 2.0x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Williams %R crosses above -80 from below AND 1d uptrend AND volume spike
            if (williams_r[i] > -80 and williams_r[i-1] <= -80 and  # Cross above -80
                supertrend_direction_aligned[i] == 1 and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Williams %R crosses below -20 from above AND 1d downtrend AND volume spike
            elif (williams_r[i] < -20 and williams_r[i-1] >= -20 and  # Cross below -20
                  supertrend_direction_aligned[i] == -1 and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R crosses opposite threshold
            # 2. 1d Supertrend direction flips
            if position == 1:
                if williams_r[i] < -20 or supertrend_direction_aligned[i] == -1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if williams_r[i] > -80 or supertrend_direction_aligned[i] == 1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals