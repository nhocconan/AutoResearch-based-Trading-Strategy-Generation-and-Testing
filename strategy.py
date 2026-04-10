#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Elder Ray + volume confirmation
# - Williams Alligator (12h): Jaw(13), Teeth(8), Lips(5) SMAs with future shifts
# - Elder Ray (1d): Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Long when: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume > 1.5x 20-period average
# - Short when: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power > 0 AND volume > 1.5x 20-period average
# - Exit when Alligator alignment breaks OR opposite signal occurs
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years)
# - Alligator identifies trend direction and strength
# - Elder Ray measures bull/bear power relative to EMA13
# - Volume confirmation reduces false signals

name = "12h_1d_alligator_elder_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Pre-compute 12h Williams Alligator
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Alligator components: SMAs with future shifts
    def sma(arr, n):
        if len(arr) < n:
            return np.full_like(arr, np.nan)
        return pd.Series(arr).rolling(window=n, min_periods=n).mean().values
    
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    jaw = sma(close, jaw_period)
    teeth = sma(close, teeth_period)
    lips = sma(close, lips_period)
    
    # Alligator lines are shifted forward by future values
    jaw_shifted = np.roll(jaw, -jaw_period//2)
    teeth_shifted = np.roll(teeth, -teeth_period//2)
    lips_shifted = np.roll(lips, -lips_period//2)
    
    # Aligned values (only valid after the shift period)
    jaw_aligned = jaw_shifted.copy()
    teeth_aligned = teeth_shifted.copy()
    lips_aligned = lips_shifted.copy()
    
    # Invalidate the shifted forward values (they represent future data)
    jaw_aligned[-jaw_period//2:] = np.nan
    teeth_aligned[-teeth_period//2:] = np.nan
    lips_aligned[-lips_period//2:] = np.nan
    
    # Pre-compute 1d Elder Ray
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high_1d - ema13_1d
    bear_power = ema13_1d - low_1d
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned_12h = align_htf_to_ltf(prices, df_1d, jaw_aligned)
    teeth_aligned_12h = align_htf_to_ltf(prices, df_1d, teeth_aligned)
    lips_aligned_12h = align_htf_to_ltf(prices, df_1d, lips_aligned)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Pre-compute 12h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned_12h[i]) or np.isnan(teeth_aligned_12h[i]) or 
            np.isnan(lips_aligned_12h[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Alligator aligned (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume spike
            if (lips_aligned_12h[i] > teeth_aligned_12h[i] > jaw_aligned_12h[i] and 
                bull_power_aligned[i] > 0 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Alligator aligned (Lips < Teeth < Jaw) AND Bear Power > 0 AND volume spike
            elif (lips_aligned_12h[i] < teeth_aligned_12h[i] < jaw_aligned_12h[i] and 
                  bear_power_aligned[i] > 0 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator alignment breaks OR opposite signal occurs
            exit_long = (position == 1 and 
                        not (lips_aligned_12h[i] > teeth_aligned_12h[i] > jaw_aligned_12h[i]))
            exit_short = (position == -1 and 
                         not (lips_aligned_12h[i] < teeth_aligned_12h[i] < jaw_aligned_12h[i]))
            
            # Also exit on opposite Elder Ray signal
            exit_long = exit_long or (position == 1 and bear_power_aligned[i] > 0)
            exit_short = exit_short or (position == -1 and bull_power_aligned[i] > 0)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals