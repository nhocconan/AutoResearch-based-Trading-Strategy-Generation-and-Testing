#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1-week EMA13 trend filter and volume spike confirmation.
# Elder Ray calculates Bull Power = High - EMA13 and Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with price > 1w EMA13 and volume spike (>2x 20-period average).
# Short when Bear Power < 0 and falling, Bull Power > 0 and rising, with price < 1w EMA13 and volume spike.
# This captures institutional buying/selling pressure while filtering with higher timeframe trend and volume confirmation.
# Designed for low trade frequency (~15-30/year) to minimize fee decay in ranging markets like 2025.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA13 trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 13-period EMA on 1w close for trend filter
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Calculate Elder Ray components on 6l data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 13-period EMA for Elder Ray (same as trend filter but on 6l)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_13_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_13_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        # Slope of power (1-bar change)
        bull_slope = bull_val - bull_power[i-1] if i > 0 else 0
        bear_slope = bear_val - bear_power[i-1] if i > 0 else 0
        
        if position == 0:
            # Long conditions: Bull Power > 0 and rising, Bear Power < 0 and falling, uptrend, volume spike
            if bull_val > 0 and bull_slope > 0 and bear_val < 0 and bear_slope < 0 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 and falling, Bull Power > 0 and rising, downtrend, volume spike
            elif bear_val < 0 and bear_slope < 0 and bull_val > 0 and bull_slope > 0 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Bull Power turns negative or Bear Power turns positive or trend breaks
                if bull_val <= 0 or bear_val >= 0 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Bear Power turns positive or Bull Power turns negative or trend breaks
                if bear_val >= 0 or bull_val <= 0 or price > ema_val:
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