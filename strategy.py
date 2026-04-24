#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d Elder Ray filter and volume confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d Williams Alligator (jaw/teeth/lips) for trend direction (bullish if lips > teeth > jaw, bearish if lips < teeth < jaw).
- HTF: 1d Elder Ray (bull power/bear power) for momentum confirmation.
- Volume: Current 12h volume > 1.5 * 20-period volume MA to avoid low-volume breakouts.
- Entry: Long when Alligator bullish AND Elder Ray bull power > 0 AND volume spike.
         Short when Alligator bearish AND Elder Ray bear power < 0 AND volume spike.
- Exit: Opposite Alligator condition OR loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams Alligator identifies trend phases via smoothed medians. Elder Ray measures bull/bear power behind price moves.
Combined with volume confirmation, this avoids false signals and works in both bull/bear markets by only taking trades
in the direction of the 1d trend with momentum behind it.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 1d (jaw=13, teeth=8, lips=5 SMAs of median price, shifted)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Median price = (high + low) / 2
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Alligator lines: SMAs of median price with specific periods and shifts
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values  # jaw: 13-period, shifted 8
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values   # teeth: 8-period, shifted 5
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values    # lips: 5-period, shifted 3
    
    # Calculate Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    df_1d_close = df_1d['close'].values
    ema_13 = pd.Series(df_1d_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 20)  # Need enough bars for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        
        # Alligator conditions: bullish if lips > teeth > jaw, bearish if lips < teeth < jaw
        alligator_bullish = lips_val > teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val < jaw_val
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Alligator bullish AND Elder Ray bull power > 0
                if alligator_bullish and bull_val > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Alligator bearish AND Elder Ray bear power < 0
                elif alligator_bearish and bear_val < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR loss of volume confirmation
            if not alligator_bullish or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR loss of volume confirmation
            if not alligator_bearish or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Williams_Alligator_1dElderRay_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0