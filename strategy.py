#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray combination
# Long when: Alligator jaws < teeth < lips (bullish alignment) AND 1d Bull Power > 0 AND 1d Bear Power < 0
# Short when: Alligator jaws > teeth > lips (bearish alignment) AND 1d Bull Power < 0 AND 1d Bear Power > 0
# Uses 6h timeframe for Alligator (Jaws=13, Teeth=8, Lips=5 SMAs smoothed) and 1d HTF for Elder Ray (EMA13)
# Discrete position sizing 0.25 to control drawdown and fee drag.
# Target: ~15-30 trades/year (50-120 over 4 years) to minimize fee drag on 6h timeframe.
# Alligator identifies trend, Elder Ray confirms bull/bear power on higher timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 6h Williams Alligator ===
    # Jaws: 13-period SMA smoothed by 8 periods
    # Teeth: 8-period SMA smoothed by 5 periods  
    # Lips: 5-period SMA smoothed by 3 periods
    jaws_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaws = pd.Series(jaws_raw).rolling(window=8, min_periods=8).mean().values
    
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # === 1d Elder Ray (Bull Power and Bear Power) ===
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Align 1d indicators to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 8, 5, 13) + 10  # Alligator + EMA13 + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Alligator bullish alignment: Jaws < Teeth < Lips
        # 2. 1d Bull Power > 0 (bulls stronger than average)
        # 3. 1d Bear Power < 0 (bears weaker than average)
        if (jaws[i] < teeth[i]) and (teeth[i] < lips[i]) and \
           (bull_power_1d_aligned[i] > 0) and (bear_power_1d_aligned[i] < 0):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Alligator bearish alignment: Jaws > Teeth > Lips
        # 2. 1d Bull Power < 0 (bulls weaker than average)
        # 3. 1d Bear Power > 0 (bears stronger than average)
        elif (jaws[i] > teeth[i]) and (teeth[i] > lips[i]) and \
             (bull_power_1d_aligned[i] < 0) and (bear_power_1d_aligned[i] > 0):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Alligator_1dElderRay_v1"
timeframe = "6h"
leverage = 1.0