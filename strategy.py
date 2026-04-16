#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d EMA13 trend filter and volume confirmation.
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d EMA13 rising AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND Bull Power > 0 AND 1d EMA13 falling AND volume > 1.5x 20-period average.
# Exit when Elder Ray signals reverse OR volume drops below average.
# Uses discrete position size 0.25. Designed to capture momentum shifts in both bull and bear markets.
# Target: 50-150 trades over 4 years (12-37/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Elder Ray Index (EMA13) ===
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13.values  # Bull Power = High - EMA13
    bear_power = low - ema13.values   # Bear Power = Low - EMA13
    
    # === 6h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 1d data once before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for EMA calculation
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 1d Indicators: EMA13 for trend filter ===
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean()
    ema13_1d_values = ema13_1d.values
    
    # Align 1d EMA13 to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d_values)
    
    # Calculate 1d EMA13 slope (rising/falling)
    ema13_slope = np.zeros_like(ema13_1d_aligned)
    ema13_slope[1:] = ema13_1d_aligned[1:] - ema13_1d_aligned[:-1]
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed)
    warmup = 30
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(ema13_slope[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema13_slope_val = ema13_slope[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Elder Ray reverses (Bear Power > 0) OR volume spike ends
            if bear_val > 0 or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Elder Ray reverses (Bull Power < 0) OR volume spike ends
            if bull_val < 0 or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power < 0 AND 1d EMA13 rising AND volume spike
            if bull_val > 0 and bear_val < 0 and ema13_slope_val > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0 AND Bull Power > 0 AND 1d EMA13 falling AND volume spike
            elif bear_val < 0 and bull_val > 0 and ema13_slope_val < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_EMA13_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0