#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Elder Ray Power with volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Elder Ray Power (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for trend strength.
- Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price.
- Entry: Long when Alligator is bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume > 1.5 * volume MA(20).
         Short when Alligator is bearish (Lips < Teeth < Jaw) AND Bear Power > 0 AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit when Alligator alignment breaks (Lips crosses Teeth).
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy combines Alligator's trend identification with Elder Ray's power measurement to catch strong trends while filtering weak moves, suitable for both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray Power (using EMA13)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d  # Bull Power = High - EMA13
    bear_power = ema13_1d - low_1d   # Bear Power = EMA13 - Low
    
    # Calculate 6h Williams Alligator components
    # Median price = (High + Low) / 2
    median_price = (high + low) / 2.0
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align HTF indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 40)  # Need enough bars for Alligator and Elder Ray
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Alligator bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND volume confirmed
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and bull_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish (Lips < Teeth < Jaw) AND Bear Power > 0 AND volume confirmed
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and bear_power_aligned[i] > 0 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Alligator alignment breaks (Lips crosses below Teeth)
            if lips[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Alligator alignment breaks (Lips crosses above Teeth)
            if lips[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dElderRay_Power_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0