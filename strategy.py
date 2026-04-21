# %%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator (13,8,5) with 1d trend filter and volume confirmation.
# In strong trends (price > 1d EMA34), Alligator signals have higher probability.
# Volume > 2x average confirms signal strength. Works in bull/bear via trend filter.
# Target: 50-150 total trades over 4 years (12-37/year). Position size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter (same timeframe)
    # Load 1d volume for confirmation
    
    # Calculate Williams Alligator lines (13,8,5 SMAs shifted)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Jaw (13-period SMMA shifted 8 bars)
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)  # Shift 8 bars forward
    jaw[:8] = np.nan  # First 8 values invalid
    
    # Teeth (8-period SMMA shifted 5 bars)
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)  # Shift 5 bars forward
    teeth[:5] = np.nan  # First 5 values invalid
    
    # Lips (5-period SMMA shifted 3 bars)
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)  # Shift 3 bars forward
    lips[:3] = np.nan  # First 3 values invalid
    
    # Align Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1-day EMA (34-period) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation using 1d volume
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume (aligned from 1d)
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, vol_1d)[i]
        
        if position == 0:
            # Enter long: Lips > Teeth > Jaw (bullish alignment) + price > 1d EMA + volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and
                price_close > ema_34_1d_aligned[i] and
                vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Lips < Teeth < Jaw (bearish alignment) + price < 1d EMA + volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and
                  price_close < ema_34_1d_aligned[i] and
                  vol_1d_current > 2.0 * vol_ma_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Alligator lines cross or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Lips < Jaw (bullish alignment broken) or trend turns down
                if (lips_aligned[i] < jaw_aligned[i]) or (price_close < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Lips > Jaw (bearish alignment broken) or trend turns up
                if (lips_aligned[i] > jaw_aligned[i]) or (price_close > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsAlligator_1dEMA34_Volume_Spike"
timeframe = "6h"
leverage = 1.0
# %%