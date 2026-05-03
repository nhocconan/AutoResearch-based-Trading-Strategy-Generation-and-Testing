#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation.
# Long: Jaw < Teeth < Lips (bullish alignment) AND price > Lips AND 1d EMA34 > 1d EMA55 AND volume > 1.3x 20-period MA
# Short: Jaw > Teeth > Lips (bearish alignment) AND price < Lips AND 1d EMA34 < 1d EMA55 AND volume > 1.3x 20-period MA
# Exit: Opposite Alligator alignment OR trend weakness (EMA34/EMA55 convergence) OR volume drop.
# Uses Williams Alligator for trend identification, 1d EMA crossover for higher timeframe trend confirmation, volume for breakout validity.
# Discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Alligator avoids whipsaws in ranging markets; 1d EMA filter ensures alignment with primary trend; volume confirms institutional participation.
# Works in bull via long signals and bear via short signals when aligned with higher timeframe trend.

name = "12h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA34 and EMA55
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema55_1d = pd.Series(close_1d).ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 1d EMA values to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    ema55_1d_aligned = align_htf_to_ltf(prices, df_1d, ema55_1d)
    
    # Williams Alligator on 12h: Jaw (13), Teeth (8), Lips (5) SMMA with offsets
    # Jaw: 13-period SMMA, shifted 8 bars forward
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMMA, shifted 5 bars forward
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMMA, shifted 3 bars forward
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Volume regime: current 12h volume > 1.3x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period for Alligator
        # Skip if any value is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema55_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema34_val = ema34_1d_aligned[i]
        ema55_val = ema55_1d_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_spike = volume_spike[i]
        
        # Determine Alligator alignment
        is_bullish_alignment = jaw_val < teeth_val < lips_val
        is_bearish_alignment = jaw_val > teeth_val > lips_val
        
        # Determine 1d trend regime
        is_uptrend = ema34_val > ema55_val
        is_downtrend = ema34_val < ema55_val
        
        # Entry logic
        if position == 0:
            # Long: Bullish Alligator alignment AND price > Lips AND 1d uptrend AND volume spike
            if is_bullish_alignment and close_val > lips_val and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment AND price < Lips AND 1d downtrend AND volume spike
            elif is_bearish_alignment and close_val < lips_val and not is_uptrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR 1d downtrend OR volume drops
            if is_bearish_alignment or not is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR 1d uptrend OR volume drops
            if is_bullish_alignment or is_uptrend or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals