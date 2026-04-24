#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d Elder Ray trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) for trend direction.
         Bullish if Bull Power > 0, bearish if Bear Power > 0.
- Volume: Current 4h volume > 1.8 * 20-period volume MA to confirm breakout strength.
- Entry: Long when Alligator Jaw < Teeth < Lips (bullish alignment) AND price > Lips AND 1d Bull Power > 0 AND volume spike.
         Short when Alligator Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND 1d Bear Power > 0 AND volume spike.
- Exit: Opposite Alligator alignment or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
- Why it should work: Alligator identifies trend inception, Elder Ray confirms 1d trend strength,
  volume filter avoids false breakouts. Works in bull (long entries) and bear (short entries).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 4h
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars  
    # Lips: 5-period SMMA, shifted 3 bars
    # SMMA = smoothed moving average (similar to EMA but with different alpha)
    jaw = pd.Series(close).ewm(alpha=1/13, adjust=False).mean().shift(8)
    teeth = pd.Series(close).ewm(alpha=1/8, adjust=False).mean().shift(5)
    lips = pd.Series(close).ewm(alpha=1/5, adjust=False).mean().shift(3)
    
    # Get 1d data for Elder Ray trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close for Elder Ray
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = df_1d['high'].values - ema_13_1d
    bear_power_1d = ema_13_1d - df_1d['low'].values
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13)  # Need enough 1d bars for EMA13 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw.iloc[i]) or np.isnan(teeth.iloc[i]) or np.isnan(lips.iloc[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        jaw_val = jaw.iloc[i]
        teeth_val = teeth.iloc[i]
        lips_val = lips.iloc[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Alligator bullish alignment (Jaw < Teeth < Lips) AND price > Lips AND 1d Bull Power > 0
                if jaw_val < teeth_val < lips_val and curr_close > lips_val and bull_power > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Alligator bearish alignment (Jaw > Teeth > Lips) AND price < Lips AND 1d Bear Power > 0
                elif jaw_val > teeth_val > lips_val and curr_close < lips_val and bear_power > 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Alligator loses bullish alignment OR loss of volume confirmation
            if not (jaw_val < teeth_val < lips_val) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator loses bearish alignment OR loss of volume confirmation
            if not (jaw_val > teeth_val > lips_val) or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dElderRay_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0