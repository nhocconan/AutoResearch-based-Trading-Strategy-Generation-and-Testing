#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation
# Long when Jaw < Teeth < Lips (bullish alignment) + 12h EMA(50) uptrend + volume spike
# Short when Jaw > Teeth > Lips (bearish alignment) + 12h EMA(50) downtrend + volume spike
# Williams Alligator uses SMAs of 13, 8, 5 periods with future offsets to avoid look-ahead
# 12h trend filter ensures alignment with higher timeframe momentum
# Volume spike confirms institutional participation
# Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag

name = "6h_WilliamsAlligator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Williams Alligator: Jaw (13-period SMMA, 8 bars ahead), Teeth (8-period, 5 bars ahead), Lips (5-period, 3 bars ahead)
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    # Using EMA as proxy for SMMA with appropriate lag compensation
    jaw_raw = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_raw = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_raw = pd.Series(close).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Apply Alligator's future offsets (Jaw: +8, Teeth: +5, Lips: +3 bars)
    # To avoid look-ahead, we use delayed values: current Jaw = jaw_raw from 8 bars ago, etc.
    jaw = np.roll(jaw_raw, 8)   # Jaw value from 8 bars ago (avoids look-ahead)
    teeth = np.roll(teeth_raw, 5) # Teeth value from 5 bars ago
    lips = np.roll(lips_raw, 3)   # Lips value from 3 bars ago
    # Fill leading NaNs from roll with first valid value
    jaw[:8] = jaw_raw[0] if not np.isnan(jaw_raw[0]) else 0
    teeth[:5] = teeth_raw[0] if not np.isnan(teeth_raw[0]) else 0
    lips[:3] = lips_raw[0] if not np.isnan(lips_raw[0]) else 0
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_12h_val = ema50_12h_aligned[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Jaw < Teeth < Lips (bullish alignment) + 12h uptrend + volume spike
            if jaw_val < teeth_val < lips_val and close[i] > ema50_12h_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Jaw > Teeth > Lips (bearish alignment) + 12h downtrend + volume spike
            elif jaw_val > teeth_val > lips_val and close[i] < ema50_12h_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alignment breaks or 12h trend turns down
            if not (jaw_val < teeth_val < lips_val) or close[i] < ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alignment breaks or 12h trend turns up
            if not (jaw_val > teeth_val > lips_val) or close[i] > ema50_12h_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals