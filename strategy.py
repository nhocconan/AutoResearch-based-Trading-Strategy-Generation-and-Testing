#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray (Bull/Bear Power) combination
# - Uses 1d Elder Ray to determine market regime (bull/bear)
# - Uses 6h Williams Alligator for entry timing within that regime
# - In bull regime (Bear Power < 0): long when Jaw < Teeth < Lips (Alligator aligned up)
# - In bear regime (Bull Power > 0): short when Jaw > Teeth > Lips (Alligator aligned down)
# - Uses discrete position sizing: ±0.25 to limit drawdown
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Williams Alligator provides trend confirmation with smoothing
# - Elder Ray filters for regime-appropriate trades only

name = "6h_1d_elder_ray_alligator_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for Elder Ray regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Compute 1d Elder Ray (Bull Power and Bear Power)
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Precompute 6h Williams Alligator components
    # Jaw: 13-period SMMA smoothed 8 bars ahead
    # Teeth: 8-period SMMA smoothed 5 bars ahead
    # Lips: 5-period SMMA smoothed 3 bars ahead
    # Using EMA as approximation for SMMA for computational efficiency
    jaw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Smooth further (SMMA effect) by applying additional EMA smoothing
    jaw = pd.Series(jaw).ewm(span=8, adjust=False, min_periods=8).mean().values
    teeth = pd.Series(teeth).ewm(span=5, adjust=False, min_periods=5).mean().values
    lips = pd.Series(lips).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine regime from Elder Ray
        # Bull regime: Bear Power < 0 (bears weak)
        # Bear regime: Bull Power > 0 (bulls weak)
        bull_regime = bear_power_aligned[i] < 0
        bear_regime = bull_power_aligned[i] > 0
        
        # Alligator alignment conditions
        # Bullish alignment: Jaw < Teeth < Lips (Alligator mouth opening up)
        # Bearish alignment: Jaw > Teeth > Lips (Alligator mouth opening down)
        bullish_aligned = jaw[i] < teeth[i] and teeth[i] < lips[i]
        bearish_aligned = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bull regime + Alligator bullish alignment
        if bull_regime and bullish_aligned:
            enter_long = True
        
        # Short: bear regime + Alligator bearish alignment
        if bear_regime and bearish_aligned:
            enter_short = True
        
        # Exit conditions: reverse of entry or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if bear regime or Alligator turns bearish
            exit_long = bear_regime or bearish_aligned
        elif position == -1:
            # Exit short if bull regime or Alligator turns bullish
            exit_short = bull_regime or bullish_aligned
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals