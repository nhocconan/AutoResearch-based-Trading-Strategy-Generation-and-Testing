#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams Alligator (13,8,5) with 4h trend filter and volume confirmation.
# In strong trends (price > 4h EMA34), Alligator signals have higher probability.
# Volume > 2x average confirms signal strength. Works in bull/bear via trend filter.
# Target: 60-150 total trades over 4 years (15-37/year). Position size: 0.20.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data for Williams Alligator calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 13:
        return np.zeros(n)
    
    # Load 4h data for EMA trend filter
    # Load 4h volume for confirmation
    
    # Calculate Williams Alligator lines (13,8,5 SMAs shifted)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Jaw (13-period SMMA shifted 8 bars)
    jaw = pd.Series(close_4h).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # Shift 8 bars forward
    jaw[:8] = np.nan  # First 8 values invalid
    
    # Teeth (8-period SMMA shifted 5 bars)
    teeth = pd.Series(close_4h).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # Shift 5 bars forward
    teeth[:5] = np.nan  # First 5 values invalid
    
    # Lips (5-period SMMA shifted 3 bars)
    lips = pd.Series(close_4h).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # Shift 3 bars forward
    lips[:3] = np.nan  # First 3 values invalid
    
    # Align Alligator lines to 1h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    
    # Calculate 4h EMA (34-period) for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation using 4h volume
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (aligned from 4h)
        price_close = prices['close'].iloc[i]
        vol_4h_current = align_htf_to_ltf(prices, df_4h, vol_4h)[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + price > 4h EMA + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                price_close > ema_34_4h_aligned[i] and
                vol_4h_current > 2.0 * vol_ma_20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + price < 4h EMA + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  price_close < ema_34_4h_aligned[i] and
                  vol_4h_current > 2.0 * vol_ma_20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: Alligator lines cross or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips < Jaw (bullish alignment broken) or trend turns down
                if (lips_aligned[i] < jaw_aligned[i]) or (price_close < ema_34_4h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Lips > Jaw (bearish alignment broken) or trend turns up
                if (lips_aligned[i] > jaw_aligned[i]) or (price_close > ema_34_4h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_WilliamsAlligator_4hEMA34_Volume_Spike"
timeframe = "1h"
leverage = 1.0