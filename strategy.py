#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA34 trend + volume confirmation
# Long when Green > Red > Blue (bullish alignment) and price > EMA34, volume spike
# Short when Blue > Red > Green (bearish alignment) and price < EMA34, volume spike
# Exit when alignment breaks or EMA34 cross reverses
# Williams Alligator uses SMAs: Jaw (13,8), Teeth (8,5), Lips (5,3)
# Designed for low trade frequency (~15-30/year) with trend-following edge
# Works in bull markets (uptrend alignment) and bear markets (downtrend alignment)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams Alligator and EMA34
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: three SMAs with different periods and shifts
    # Jaw: 13-period SMA, shifted 8 bars forward
    jaw_raw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMA, shifted 5 bars forward
    teeth_raw = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMA, shifted 3 bars forward
    lips_raw = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # 34-period EMA for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_34_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) and price > EMA34, volume spike
            if lips_val > teeth_val and teeth_val > jaw_val and price > ema_34_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Jaw > Teeth > Lips (bearish alignment) and price < EMA34, volume spike
            elif jaw_val > teeth_val and teeth_val > lips_val and price < ema_34_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: alignment breaks or EMA34 cross reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when bullish alignment breaks or price crosses below EMA34
                if not (lips_val > teeth_val and teeth_val > jaw_val) or price < ema_34_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when bearish alignment breaks or price crosses above EMA34
                if not (jaw_val > teeth_val and teeth_val > lips_val) or price > ema_34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0