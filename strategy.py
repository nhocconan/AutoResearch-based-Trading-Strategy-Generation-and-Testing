#!/usr/bin/env python3
# Hypothesis: 6h Chaikin Money Flow (CMF) + 1d Supertrend + volume confirmation
# Long when: CMF > 0.15 (bullish money flow), 1d Supertrend = uptrend, volume > 1.3x 20-period average
# Short when: CMF < -0.15 (bearish money flow), 1d Supertrend = downtrend, volume > 1.3x 20-period average
# Exit when: CMF crosses back to zero OR Supertrend flips
# Position size: 0.25. Target: 20-40 trades/year to avoid fee drag.
# Works in bull (strong CMF + trend) and bear (strong negative CMF + trend) via short signals.

name = "6h_CMF_1dSupertrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Chaikin Money Flow (CMF) - 20 period
    # CMF = sum((close - low - (high - close)) / (high - low) * volume) / sum(volume)
    # Avoid division by zero
    hl_range = high - low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)  # prevent div by zero
    mfm = ((close - low) - (high - close)) / hl_range  # Money Flow Multiplier
    mfv = mfm * volume  # Money Flow Volume
    
    # 20-period sums
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = mfv_sum / vol_sum  # Chaikin Money Flow
    
    # Get daily data for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Supertrend calculation (10, 3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Upper and Lower Bands
    hl_avg = (df_1d['high'] + df_1d['low']) / 2
    upper_band = hl_avg + (multiplier * atr)
    lower_band = hl_avg - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros(len(df_1d))
    supertrend_upt = True  # True = uptrend
    
    for i in range(atr_period, len(df_1d)):
        if i == atr_period:
            supertrend[i] = upper_band.iloc[i] if close.iloc[i] > upper_band.iloc[i] else lower_band.iloc[i]
            supertrend_upt = close.iloc[i] > supertrend[i]
        else:
            if supertrend_upt:
                supertrend[i] = lower_band.iloc[i]
                if close.iloc[i] <= supertrend[i]:
                    supertrend_upt = False
                    supertrend[i] = upper_band.iloc[i]
                else:
                    supertrend[i] = max(supertrend[i-1], lower_band.iloc[i])
            else:
                supertrend[i] = upper_band.iloc[i]
                if close.iloc[i] >= supertrend[i]:
                    supertrend_upt = True
                    supertrend[i] = lower_band.iloc[i]
                else:
                    supertrend[i] = min(supertrend[i-1], upper_band.iloc[i])
    
    # Align Supertrend trend (True = uptrend, False = downtrend)
    supertrend_upt_aligned = align_htf_to_ltf(prices, df_1d, supertrend_upt)
    
    # Volume spike: current volume > 1.3x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.3 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for CMF and Supertrend
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(cmf[i]) or np.isnan(supertrend_upt_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: CMF > 0.15, Supertrend uptrend, volume spike
            if (cmf[i] > 0.15 and 
                supertrend_upt_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: CMF < -0.15, Supertrend downtrend, volume spike
            elif (cmf[i] < -0.15 and 
                  not supertrend_upt_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CMF crosses below 0 OR Supertrend flips down
            if (cmf[i] < 0) or (not supertrend_upt_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CMF crosses above 0 OR Supertrend flips up
            if (cmf[i] > 0) or (supertrend_upt_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals