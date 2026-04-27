#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation.
# Williams Alligator uses three SMAs (Jaw=13, Teeth=8, Lips=5) to identify market phases.
# When all three lines are intertwined (no clear separation), market is sleeping (range).
# When Jaw > Teeth > Lips (down) or Lips > Teeth > Jaw (up), market is awake (trend).
# Strategy: Enter long when Lips crosses above Teeth and Jaw in uptrend (1d EMA > price).
# Enter short when Lips crosses below Teeth and Jaw in downtrend (1d EMA < price).
# Volume spike confirms participation. Designed for ~25-35 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: three smoothed SMAs
    # Jaw: 13-period SMMA, 8 periods ahead
    # Teeth: 8-period SMMA, 5 periods ahead  
    # Lips: 5-period SMMA, 3 periods ahead
    def smma(data, period):
        sma = pd.Series(data).rolling(window=period, min_periods=period).mean().values
        # SMMA: first value = SMA, then SMMA = (prev*(period-1) + current) / period
        smma_vals = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            smma_vals[period-1] = sma[period-1]
            for i in range(period, len(data)):
                if not np.isnan(smma_vals[i-1]):
                    smma_vals[i] = (smma_vals[i-1] * (period-1) + data[i]) / period
                else:
                    smma_vals[i] = np.nan
        return smma_vals
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift to avoid look-ahead (SMMA already includes smoothing)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values become NaN after roll
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # 34-period EMA on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator sleeping (intertwined) = no trade
        # Check if lines are separated enough
        jaw_teeth_sep = abs(jaw[i] - teeth[i]) > (close[i] * 0.001)
        teeth_lips_sep = abs(teeth[i] - lips[i]) > (close[i] * 0.001)
        lips_jaw_sep = abs(lips[i] - jaw[i]) > (close[i] * 0.001)
        
        # Market awake when at least two separations exist
        awake = (jaw_teeth_sep + teeth_lips_sep + lips_jaw_sep) >= 2
        
        if awake:
            # Determine trend direction from Alligator alignment
            # Uptrend: Lips > Teeth > Jaw
            # Downtrend: Jaw > Teeth > Lips
            if lips[i] > teeth[i] and teeth[i] > jaw[i]:  # Uptrend
                if close[i] > ema34_1d_aligned[i] and volume_filter[i]:  # Only long in uptrend vs 1d EMA
                    # Enter on Lips crossing above Teeth (already true if we're here)
                    # But require pullback to avoid chasing
                    if close[i] <= lips[i] * 1.005 and close[i] >= lips[i] * 0.995:
                        signals[i] = 0.25
                        position = 1
            elif jaw[i] > teeth[i] and teeth[i] > lips[i]:  # Downtrend
                if close[i] < ema34_1d_aligned[i] and volume_filter[i]:  # Only short in downtrend vs 1d EMA
                    # Enter on Lips crossing below Teeth
                    if close[i] >= teeth[i] * 0.995 and close[i] <= teeth[i] * 1.005:
                        signals[i] = -0.25
                        position = -1
            else:
                # Mixed signals - hold or flatten
                if position == 1 and close[i] < jaw[i]:  # Exit long if price crosses below Jaw
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] > jaw[i]:  # Exit short if price crosses above Jaw
                    signals[i] = 0.0
                    position = 0
                else:
                        signals[i] = 0.0
        else:
            # Market sleeping - no new entries, hold existing if profitable
            if position == 1 and close[i] > lips[i]:  # Hold long while above Lips
                signals[i] = 0.25
            elif position == -1 and close[i] < lips[i]:  # Hold short while below Lips
                signals[i] = -0.25
            else:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0