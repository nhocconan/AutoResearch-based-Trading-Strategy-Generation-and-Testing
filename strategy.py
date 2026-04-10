#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray power with 1d volume regime filter
# - Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when Bull Power > 0 AND Bear Power < 0 (bulls in control) AND 1d volume > 1.2x 20-period average (strong participation)
# - Short when Bull Power < 0 AND Bear Power > 0 (bears in control) AND 1d volume > 1.2x 20-period average
# - Exit when power diverges (Bull Power < 0 for longs, Bear Power < 0 for shorts) OR volume drops below average
# - Volume regime filter ensures trades occur with institutional participation, reducing false signals
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_elderray_volume_regime_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power_1d = ema13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Align HTF Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF volume average to 6h timeframe
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume regime condition: current 1d volume > 1.2x 20-period average (strong participation)
        vol_1d_current = df_1d['volume'].values
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d_current)
        strong_volume = not np.isnan(vol_1d_aligned[i]) and not np.isnan(vol_ma_20_aligned[i]) and \
                        vol_1d_aligned[i] > 1.2 * vol_ma_20_aligned[i]
        
        bull_now = bull_power_aligned[i]
        bear_now = bear_power_aligned[i]
        
        # Elder Ray signals
        long_signal = bull_now > 0 and bear_now < 0  # Bulls in control
        short_signal = bull_now < 0 and bear_now > 0  # Bears in control
        exit_long = bull_now < 0  # Long exit: bulls lose control
        exit_short = bear_now < 0  # Short exit: bears lose control
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bulls in control AND strong volume
            if long_signal and strong_volume:
                position = 1
                signals[i] = 0.25
            # Short conditions: bears in control AND strong volume
            elif short_signal and strong_volume:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: power diverges OR volume drops below average
            vol_exhaustion = not np.isnan(vol_1d_aligned[i]) and not np.isnan(vol_ma_20_aligned[i]) and \
                           vol_1d_aligned[i] < vol_ma_20_aligned[i]
            exit_condition = (position == 1 and exit_long) or (position == -1 and exit_short) or vol_exhaustion
            
            if exit_condition:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals