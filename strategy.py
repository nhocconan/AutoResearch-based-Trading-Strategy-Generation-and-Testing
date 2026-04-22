#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1w EMA13 trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 on 1w timeframe.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume spike (>2x 20-period avg).
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with volume spike.
# Uses 1w trend to filter trades in both bull and bear markets, reducing false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for Elder Ray calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 13-period EMA on 1w close
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1w - ema_13_1w
    bear_power = low_1w - ema_13_1w
    
    # Align 1w indicators to 6h timeframe (waits for 1w bar to close)
    bull_power_aligned = align_htf_to_ltf(prices, df_1w, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1w, bear_power)
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema_13_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        ema_val = ema_13_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: Bull Power > 0 and rising, Bear Power < 0 and falling, volume spike
            if bull_val > 0 and bull_val > bull_power_aligned[i-1] and bear_val < 0 and bear_val < bear_power_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 and falling, Bull Power < 0 and rising, volume spike
            elif bear_val < 0 and bear_val < bear_power_aligned[i-1] and bull_val < 0 and bull_val > bull_power_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse of entry
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power <= 0 or Bear Power >= 0
                if bull_val <= 0 or bear_val >= 0:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power >= 0 or Bull Power <= 0
                if bear_val >= 0 or bull_val <= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1wEMA13_Volume"
timeframe = "6h"
leverage = 1.0