#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA200 trend filter and volume confirmation.
# Long when Bull Power > 0, price above 1d EMA200, and volume > 1.5x 6s average volume.
# Short when Bear Power < 0, price below 1d EMA200, and volume > 1.5x 6s average volume.
# Exit when Bull/Bear Power crosses zero.
# Uses Elder Ray for momentum, EMA200 for trend filter, volume for confirmation.
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "6h_ElderRay_EMA200_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 and EMA200 (1d) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Bull Power = High - EMA13
    bull_power = high - ema13
    # Bear Power = Low - EMA13
    bear_power = low - ema13
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Get 6s average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bp = bull_power[i]
        br = bear_power[i]
        ema200_val = ema200_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: Bull Power > 0, price above 1d EMA200, volume confirmation
            if bp > 0 and price > ema200_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power < 0, price below 1d EMA200, volume confirmation
            elif br < 0 and price < ema200_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power crosses below zero
            if bp <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power crosses above zero
            if br >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals