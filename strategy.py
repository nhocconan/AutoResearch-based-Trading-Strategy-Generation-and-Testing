#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with EMA200 filter and volume confirmation.
# Bull Power = High - EMA(200), Bear Power = EMA(200) - Low.
# Long when Bull Power > 0 and increasing, price > EMA200, volume > 1.5x average.
# Short when Bear Power > 0 and increasing, price < EMA200, volume > 1.5x average.
# Exit when power decreases or price crosses EMA200.
# Uses 6h timeframe with daily EMA200 for trend filter.
# Target: 12-30 trades/year per symbol to stay within frequency limits.
name = "6h_ElderRay_EMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA200 calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(200) on daily close
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA200 to 6h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate EMA(20) for Elder Ray on 6h timeframe
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema_20
    bear_power = ema_20 - low
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema200 = ema_200_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # Calculate power changes (only if we have previous values)
        if i > start_idx:
            bp_change = bp - bull_power[i-1]
            br_change = br - bear_power[i-1]
        else:
            bp_change = 0
            br_change = 0
        
        if position == 0:
            # Long entry: Bull Power > 0 and rising, price > EMA200, volume confirmation
            if bp > 0 and bp_change > 0 and price > ema200 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power > 0 and rising, price < EMA200, volume confirmation
            elif br > 0 and br_change > 0 and price < ema200 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power decreases or price crosses below EMA200
            if bp_change < 0 or price < ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power decreases or price crosses above EMA200
            if br_change < 0 or price > ema200:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals