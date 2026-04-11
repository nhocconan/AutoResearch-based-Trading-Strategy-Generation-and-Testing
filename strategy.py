#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_williams_alligator_v1"
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
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Williams Alligator from daily data
    # Jaw (blue): 13-period SMMA, shifted 8 bars forward
    # Teeth (red): 8-period SMMA, shifted 5 bars forward
    # Lips (green): 5-period SMMA, shifted 3 bars forward
    close_1d = df_1d['close'].values
    
    # Calculate SMMA (Smoothed Moving Average)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (Prev SMMA*(period-1) + Current Price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    smma5 = smma(close_1d, 5)
    smma8 = smma(close_1d, 8)
    smma13 = smma(close_1d, 13)
    
    # Apply shifts (forward shift means future values, so we need to delay for alignment)
    # Jaw: 13-period SMMA shifted 8 bars forward -> use value from 8 bars ago
    jaw = np.roll(smma13, 8)
    jaw[:8] = np.nan
    # Teeth: 8-period SMMA shifted 5 bars forward
    teeth = np.roll(smma8, 5)
    teeth[:5] = np.nan
    # Lips: 5-period SMMA shifted 3 bars forward
    lips = np.roll(smma5, 3)
    lips[:3] = np.nan
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_current > 1.5 * vol_ma_20[i]
        
        # Alligator conditions
        # Bullish: Lips > Teeth > Jaw (all aligned and in order)
        bullish = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        # Bearish: Jaws > Teeth > Lips (all aligned and in order)
        bearish = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Bullish alignment with volume confirmation
        if bullish and vol_confirm:
            enter_long = True
        
        # Short: Bearish alignment with volume confirmation
        if bearish and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite alignment
        exit_long = bearish  # Exit long when bearish alignment forms
        exit_short = bullish  # Exit short when bullish alignment forms
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 6s Williams Alligator strategy with volume confirmation.
# Uses Williams Alligator (Smoothed Moving Averages) on daily timeframe to identify trend.
# Enters long when Lips > Teeth > Jaw (bullish alignment) with volume > 1.5x 20-period average.
# Enters short when Jaw > Teeth > Lips (bearish alignment) with volume confirmation.
# Exits when opposite alignment occurs.
# Williams Alligator is effective in both trending and ranging markets - in ranging markets,
# the lines intertwine, reducing false signals; in strong trends, they separate clearly.
# The 6h timeframe provides good balance between signal quality and trade frequency.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Position size 0.25 manages risk while allowing meaningful returns.