#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1d Williams Alligator for trend direction and 6h momentum confirmation
# Williams Alligator (JAWS=13, TEETH=8, LIPS=5) identifies trend via alignment of SMAs
# In trending markets: JAWS > TEETH > LIPS (downtrend) or JAWS < TEETH < LIPS (uptrend)
# Entry confirmed by 6h price crossing above/below 8-period SMA with volume surge
# Works in both bull and bear markets: Alligator filters choppy markets, momentum triggers capture trend continuations

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Williams Alligator
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator on 1d data
    # JAWS: 13-period SMMA shifted 8 bars
    # TEETH: 8-period SMMA shifted 5 bars
    # LIPS: 5-period SMMA shifted 3 bars
    # Using SMA as approximation for SMMA (simple moving average)
    close_1d = df_1d['close'].values
    
    # Calculate SMAs
    sma_5 = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    sma_8 = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    sma_13 = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    
    # Shift for Alligator lines
    lips = np.concatenate([np.full(3, np.nan), sma_5[:-3]])   # 5-period SMA shifted 3
    teeth = np.concatenate([np.full(5, np.nan), sma_8[:-5]])  # 8-period SMA shifted 5
    jaws = np.concatenate([np.full(8, np.nan), sma_13[:-8]])  # 13-period SMA shifted 8
    
    # Align Alligator lines to 6h timeframe
    jaws_aligned = align_htf_to_ltf(prices, df_1d, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 6h 8-period SMA for momentum confirmation
    sma_8_6h = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    
    # Calculate 6h volume average for surge detection
    vol_avg_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # Ensure we have enough data for all indicators
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(sma_8_6h[i]) or
            np.isnan(vol_avg_6h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Williams Alligator trend detection
        # Uptrend: JAWS < TEETH < LIPS
        # Downtrend: JAWS > TEETH > LIPS
        # Avoid trading when intertwined (choppy market)
        jaws_val = jaws_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        
        # Check for clear trend separation
        uptrend = jaws_val < teeth_val < lips_val
        downtrend = jaws_val > teeth_val > lips_val
        
        # Volume surge confirmation (current volume > 1.5x average)
        volume_surge = vol > 1.5 * vol_avg_6h[i]
        
        if position == 0:
            # Enter long: uptrend + price above SMA8 + volume surge
            if uptrend and price > sma_8_6h[i] and volume_surge:
                position = 1
                signals[i] = position_size
            # Enter short: downtrend + price below SMA8 + volume surge
            elif downtrend and price < sma_8_6h[i] and volume_surge:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend weakening (Alligator lines converging) or price crosses below SMA8
            # Convergence: JAWS > TEETH or TEETH > LIPS (uptrend breaking down)
            trend_weakening = jaws_val > teeth_val or teeth_val > lips_val
            price_below_sma = price < sma_8_6h[i]
            
            if trend_weakening or price_below_sma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend weakening (Alligator lines converging) or price crosses above SMA8
            # Convergence: JAWS < TEETH or TEETH < LIPS (downtrend breaking down)
            trend_weakening = jaws_val < teeth_val or teeth_val < lips_val
            price_above_sma = price > sma_8_6h[i]
            
            if trend_weakening or price_above_sma:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1dWilliamsAlligator_MomentumSurge_v1"
timeframe = "6h"
leverage = 1.0