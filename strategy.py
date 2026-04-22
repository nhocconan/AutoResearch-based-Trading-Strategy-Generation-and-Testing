#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h trend filter and volume confirmation.
# Long when green line (lips) > red line (teeth) > blue line (jaw) + volume spike + price > 12h EMA50
# Short when green line < red line < blue line + volume spike + price < 12h EMA50
# Exit when Alligator lines re-converge (lips crosses teeth) or volume drops below 70% of average.
# Alligator is trend-following; works in strong trends (bull/bear). Volume filters false breakouts.
# Target: 20-35 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Alligator and EMA50
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams Alligator: SMAs of median price
    # Jaw (blue): 13-period SMA, shifted 8 bars
    # Teeth (red): 8-period SMA, shifted 5 bars
    # Lips (green): 5-period SMA, shifted 3 bars
    median_price = (high_12h + low_12h) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike filter (24-period average)
    volume = prices['volume'].values
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema50 = ema50_aligned[i]
        
        # Volume filter: current volume > 1.7 * 24-period average
        vol_spike = vol > 1.7 * vol_ma
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (Alligator awake, eating up) + volume spike + price > EMA50
            if lips_val > teeth_val and teeth_val > jaw_val and vol_spike and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (Alligator awake, eating down) + volume spike + price < EMA50
            elif lips_val < teeth_val and teeth_val < jaw_val and vol_spike and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: Alligator re-converges (lips crosses teeth) or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when lips crosses below teeth (trend weakening) or volume dries up
                if lips_val < teeth_val or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when lips crosses above teeth (trend weakening) or volume dries up
                if lips_val > teeth_val or vol < 0.7 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Williams_Alligator_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0