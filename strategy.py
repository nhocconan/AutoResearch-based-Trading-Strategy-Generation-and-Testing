#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Williams Alligator (Jaw/Teeth/Lips) with 1w trend filter (EMA50) and volume confirmation.
# Long when Lips > Teeth > Jaw (bullish alignment), price > 1w EMA50, and 12h volume > 1.5x 20-period median volume.
# Short when Lips < Teeth < Jaw (bearish alignment), price < 1w EMA50, and same volume condition.
# Exit when Alligator alignment breaks (Lips crosses Teeth or Teeth crosses Jaw).
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# The Alligator identifies trending vs ranging markets via convergence/divergence of its three SMAs.
# Combined with 1w EMA50 regime filter and volume spike, it avoids whipsaws and captures strong trends in both bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Alligator and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Williams Alligator (Jaw, Teeth, Lips) and volume median ===
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # SMMA = Smoothed Moving Average (similar to EMA but with different smoothing)
    jaw_12h = pd.Series(close_12h).ewm(span=13, adjust=False, min_periods=13).mean().values
    teeth_12h = pd.Series(close_12h).ewm(span=8, adjust=False, min_periods=8).mean().values
    lips_12h = pd.Series(close_12h).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate 12h volume median (20-period)
    vol_median_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).median().values
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA50 for trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_12h)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_12h)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_12h)
    vol_median_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20)
    vol_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_12h)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 13, 8, 5, 50)  # volume median(20), Jaw(13), Teeth(8), Lips(5), EMA50(1w)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(vol_12h_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        jaw = jaw_aligned[i]
        teeth = teeth_aligned[i]
        lips = lips_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_12h = vol_12h_aligned[i]
        ema_50_1w = ema_50_1w_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when bullish alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if lips <= teeth or teeth <= jaw:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when bearish alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if lips >= teeth or teeth >= jaw:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 12h volume > 1.5x median volume
            volume_spike = vol_12h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Bullish alignment: Lips > Teeth > Jaw, price > 1w EMA50 (uptrend regime), and volume spike
            if lips > teeth and teeth > jaw and close[i] > ema_50_1w and volume_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT CONDITIONS
            # Bearish alignment: Lips < Teeth < Jaw, price < 1w EMA50 (downtrend regime), and volume spike
            elif lips < teeth and teeth < jaw and close[i] < ema_50_1w and volume_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_WilliamsAlligator_1wEMA50_12hVolumeSpike1.5x_v1"
timeframe = "12h"
leverage = 1.0