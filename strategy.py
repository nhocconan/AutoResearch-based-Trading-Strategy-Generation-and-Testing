#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_alligator_ema50_v1"
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
    
    # Calculate 1d Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Jaw: 13-period SMMA (shifted 8)
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(sma_13, 8)
    
    # Teeth: 8-period SMMA (shifted 5)
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(sma_8, 5)
    
    # Lips: 5-period SMMA (shifted 3)
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(sma_5, 3)
    
    # Shift by 1 to use only completed 1d bars
    jaw = np.roll(jaw, 1)
    teeth = np.roll(teeth, 1)
    lips = np.roll(lips, 1)
    jaw[0] = np.nan
    teeth[0] = np.nan
    lips[0] = np.nan
    
    # Align 1d Alligator lines to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 6h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(60, n):  # Start after EMA50 warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        bullish_alignment = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alignment = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # Long conditions: Bullish alignment AND price > EMA50 with volume
        long_signal = volume_confirmed and bullish_alignment and (price_close > ema_50[i])
        
        # Short conditions: Bearish alignment AND price < EMA50 with volume
        short_signal = volume_confirmed and bearish_alignment and (price_close < ema_50[i])
        
        # Exit when Alligator lines cross (Lips crosses Teeth)
        exit_long = position == 1 and lips_aligned[i] <= teeth_aligned[i]
        exit_short = position == -1 and lips_aligned[i] >= teeth_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
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

# Hypothesis: Williams Alligator + EMA50 trend filter on 6h with volume confirmation.
# Uses 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5 SMAs) to identify trend direction
# and alignment. Enters long when Lips > Teeth > Jaw (bullish alignment) and price
# above 6h EMA50 with volume confirmation (>1.8x average). Enters short when
# Lips < Teeth < Jaw (bearish alignment) and price below 6h EMA50 with volume confirmation.
# Exits when Lips crosses Teeth (trend weakening). Alligator acts as a dynamic trend
# filter that avoids whipsaws in ranging markets. Volume ensures participation.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drift on 6h.