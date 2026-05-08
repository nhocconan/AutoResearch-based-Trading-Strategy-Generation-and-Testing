#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d trend filter and volume confirmation.
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13).
# In bull regime (price > 1d EMA50): enter long when Bull Power turns positive after being negative.
# In bear regime (price < 1d EMA50): enter short when Bear Power turns negative after being positive.
# Volume spike confirms institutional participation. Designed to catch trend reversals in both bull and bear markets.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_ElderRay_Reversal_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA13 for Elder Ray (on 6x data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Sufficient warmup for EMA13
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: bull if close > 1d EMA50, bear if close < 1d EMA50
            is_bull_regime = close[i] > ema_50_1d_aligned[i]
            is_bear_regime = close[i] < ema_50_1d_aligned[i]
            
            # Long: bull regime + Bull Power turns positive (was negative or zero) + volume spike
            if is_bull_regime:
                bull_power_prev = bull_power[i-1]
                bull_power_cross_up = (bull_power[i] > 0) and (bull_power_prev <= 0)
                if bull_power_cross_up and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
            
            # Short: bear regime + Bear Power turns negative (was positive or zero) + volume spike
            elif is_bear_regime:
                bear_power_prev = bear_power[i-1]
                bear_power_cross_down = (bear_power[i] < 0) and (bear_power_prev >= 0)
                if bear_power_cross_down and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: Bear Power turns negative (momentum loss) or reversal signal
            if bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bull Power turns positive (momentum loss) or reversal signal
            if bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals