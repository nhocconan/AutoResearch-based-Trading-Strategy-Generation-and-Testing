#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period high and closes above 12h EMA50 with volume > 1.5x 4h average volume.
# Short when price breaks below 20-period low and closes below 12h EMA50 with volume > 1.5x 4h average volume.
# Exit when price reverses and closes below/above 12h EMA50.
# Uses Donchian for breakout, EMA for trend filter, volume for confirmation.
# Target: 20-50 trades/year per symbol to stay within frequency limits.
name = "4h_Donchian20_EMA50_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        don_high = donchian_high[i]
        don_low = donchian_low[i]
        ema_50 = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price breaks above Donchian high, closes above EMA50, with volume confirmation
            if high[i] > don_high and close[i] > ema_50 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low, closes below EMA50, with volume confirmation
            elif low[i] < don_low and close[i] < ema_50 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below EMA50
            if close[i] < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above EMA50
            if close[i] > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals