#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter
# - Primary: 6h Elder Ray (Bull Power = EMA13(high) - EMA13(close), Bear Power = EMA13(low) - EMA13(close))
# - Regime filter: 1d Williams %R > -20 (overbought) for shorts, < -80 (oversold) for longs
# - Entry: Long when Bull Power > 0 and Bear Power < 0 and Williams %R < -80
#          Short when Bear Power < 0 and Bull Power > 0 and Williams %R > -20
# - Exit: Opposite signal or Elder Ray divergence (Bull Power < 0 for long exit, Bear Power > 0 for short exit)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Elder Ray shows power balance, Williams %R avoids extremes, regime-adaptive

name = "6h_1d_elder_ray_williams_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray on 6h timeframe
    ema13_close = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_high = pd.Series(high).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_low = pd.Series(low).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = ema13_high - ema13_close  # EMA13(high) - EMA13(close)
    bear_power = ema13_low - ema13_close   # EMA13(low) - EMA13(close)
    
    # Calculate Williams %R on 1d timeframe
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(williams_r_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 (bulls in control), Bear Power < 0 (bears weak), Williams %R oversold (< -80)
            if (bull_power[i] > 0 and bear_power[i] < 0 and williams_r_aligned[i] < -80):
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 (bears in control), Bull Power > 0 (bulls weak), Williams %R overbought (> -20)
            elif (bear_power[i] < 0 and bull_power[i] > 0 and williams_r_aligned[i] > -20):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions:
            # Long exit: Bull Power <= 0 (bulls losing control) OR Bear Power >= 0 (bears gaining control)
            # Short exit: Bear Power >= 0 (bears losing control) OR Bull Power <= 0 (bulls gaining control)
            if position == 1:  # Long position
                if bull_power[i] <= 0 or bear_power[i] >= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if bear_power[i] >= 0 or bull_power[i] <= 0:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals