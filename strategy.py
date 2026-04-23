#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1w Camarilla Pivot Breakout
- Williams Alligator (Jaw/Teeth/Lips) identifies trend absence (alligator sleeping) vs presence (awake)
- 1w Camarilla levels provide institutional support/resistance from weekly structure
- Long when price breaks above weekly R3 with alligator awake (Lips > Teeth > Jaw)
- Short when price breaks below weekly S3 with alligator awake (Lips < Teeth < Jaw)
- Volume confirmation > 2.0x 24-period average ensures institutional participation
- Designed for low-frequency, high-conviction trades: target 12-37/year (50-150 over 4 years)
- Works in bull/bear markets by trading breakouts of weekly structure with trend confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for weekly Camarilla pivots
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels (based on prior week OHLC)
    # Using typical Camarilla formula: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # where C = (H+L+O)/3 (typical price)
    typical_1w = (df_1w['high'] + df_1w['low'] + df_1w['open']) / 3
    range_1w = df_1w['high'] - df_1w['low']
    
    # Weekly Camarilla levels
    camarilla_r3_1w = typical_1w + (range_1w * 1.1 / 4)
    camarilla_s3_1w = typical_1w - (range_1w * 1.1 / 4)
    camarilla_r4_1w = typical_1w + (range_1w * 1.1 / 2)
    camarilla_s4_1w = typical_1w - (range_1w * 1.1 / 2)
    
    # Align weekly levels to 6h timeframe (wait for weekly bar to close)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3_1w.values)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3_1w.values)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w.values)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w.values)
    
    # Williams Alligator on 6h timeframe
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    def smma(series, period):
        """Smoothed Moving Average"""
        if len(series) < period:
            return np.full_like(series, np.nan)
        result = np.full_like(series, np.nan)
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        result[period-1] = sma[period-1]
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift jaws forward (Alligator lines are plotted forward)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set initial shifted values to NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: > 2.0x 24-period average (6h * 24 = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 13, 8, 5) + 8  # Account for Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(s3_1w_aligned[i]) or
            np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator awake condition: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
        alligator_awake_up = (lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i])
        alligator_awake_down = (lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i])
        
        # Breakout signals with volume confirmation
        long_breakout = (close[i] > r3_1w_aligned[i] and volume[i] > 2.0 * vol_ma[i])
        short_breakout = (close[i] < s3_1w_aligned[i] and volume[i] > 2.0 * vol_ma[i])
        
        # Strong breakout confirmation (beyond R4/S4 adds momentum)
        strong_long = long_breakout and (close[i] > r4_1w_aligned[i])
        strong_short = short_breakout and (close[i] < s4_1w_aligned[i])
        
        if position == 0:
            # Enter only when alligator is awake and breakout occurs
            if alligator_awake_up and strong_long:
                signals[i] = 0.25
                position = 1
            elif alligator_awake_down and strong_short:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator sleeping (trend weakening) or opposite breakout
            exit_signal = False
            
            if position == 1:
                # Exit long: alligator goes to sleep or short breakout
                if not alligator_awake_up or (close[i] < s3_1w_aligned[i] and volume[i] > 2.0 * vol_ma[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: alligator goes to sleep or long breakout
                if not alligator_awake_down or (close[i] > r3_1w_aligned[i] and volume[i] > 2.0 * vol_ma[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1wCamarillaR3S3_Breakout_VolumeConfirm"
timeframe = "6h"
leverage = 1.0