#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d Elder Ray Bear Power + Volume Spike.
- Primary timeframe: 12h for signal generation.
- HTF: 1d for Elder Ray Bear Power (trend filter: bearish when Bear Power < 0 AND declining).
- Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs on median price. Bullish when Lips > Teeth > Jaw.
- Volume: Current 12h volume > 2.0 * 20-period volume MA to avoid false signals.
- Entry: Long when Alligator bullish AND Bear Power < 0 AND rising (momentum shift) AND volume spike.
         Short when Alligator bearish AND Bear Power > 0 AND falling AND volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Why works in bull/bear: Alligator identifies trend, Elder Ray measures bull/bear power, volume confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate median price for Alligator
    median_price = (high + low) / 2.0
    
    # Calculate Williams Alligator on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    median_price_12h = (df_12h['high'].values + df_12h['low'].values) / 2.0
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values    # 8-period SMA
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values     # 5-period SMA
    
    # Align Alligator components to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate Elder Ray Bear Power on 1d: Bear Power = Low - EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    ema_13 = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    bear_power = df_1d['low'].values - ema_13  # Bear Power = Low - EMA13
    
    # Align Bear Power to 12h
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 20-period volume MA on 12h
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 12h volume MA
    volume_spike = volume > (2.0 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough bars for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bear_power_val = bear_power_aligned[i]
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Lips > Teeth > Jaw (Alligator aligned up) AND Bear Power < 0 AND rising (bullish momentum)
                if lips_val > teeth_val > jaw_val and bear_power_val < 0 and bear_power_val > bear_power_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Lips < Teeth < Jaw (Alligator aligned down) AND Bear Power > 0 AND falling (bearish momentum)
                elif lips_val < teeth_val < jaw_val and bear_power_val > 0 and bear_power_val < bear_power_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR loss of volume confirmation
            if not (lips_val > teeth_val > jaw_val) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR loss of volume confirmation
            if not (lips_val < teeth_val < jaw_val) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0