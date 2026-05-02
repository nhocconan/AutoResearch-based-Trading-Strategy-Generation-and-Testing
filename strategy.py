#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray + volume spike
# Williams Alligator (JAWS=13, TEETH=8, LIPS=5 SMMA) identifies trend alignment
# Elder Ray (Bull/Bear Power = Close - EMA13) measures trend strength from 1d
# Volume spike (>2.0 x 20 EMA) confirms breakout validity
# Works in bull markets (Alligator bullish + Elder Ray bullish + volume) and bear markets (Alligator bearish + Elder Ray bearish + volume)
# Uses discrete position sizing (0.25) to balance return and drawdown control
# Target: 50-150 total trades over 4 years = 12-37/year

name = "6h_WilliamsAlligator_1dElderRay_VolumeSpike"
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
    
    # 6h data for Williams Alligator (SMMA)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Smoothed Moving Average (SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    close_6h = df_6h['close'].values
    jaws = smma(close_6h, 13)  # Blue line
    teeth = smma(close_6h, 8)   # Red line
    lips = smma(close_6h, 5)    # Green line
    
    jaws_aligned = align_htf_to_ltf(prices, df_6h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips)
    
    # 1d data for Elder Ray (Bull/Bear Power = Close - EMA13)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA13 on 1d close
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = df_1d['close'].values - ema_13_1d
    bear_power = ema_13_1d - df_1d['close'].values
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Alligator and EMA calculation)
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine Alligator trend: 
        # Bullish: Lips > Teeth > Jaws (Green > Red > Blue)
        # Bearish: Jaws > Teeth > Lips (Blue > Red > Green)
        alligator_bullish = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaws_aligned[i]
        alligator_bearish = jaws_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]
        
        # Determine Elder Ray trend from 1d
        elder_bullish = bull_power_aligned[i] > 0  # Close > EMA13
        elder_bearish = bear_power_aligned[i] > 0  # EMA13 > Close (equivalent to bull_power < 0)
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish + Elder Ray bullish + volume confirmation
            if alligator_bullish and elder_bullish and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish + Elder Ray bearish + volume confirmation
            elif alligator_bearish and elder_bearish and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Elder Ray turns bearish
            if not alligator_bullish or not elder_bullish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Elder Ray turns bullish
            if not alligator_bearish or not elder_bearish:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals