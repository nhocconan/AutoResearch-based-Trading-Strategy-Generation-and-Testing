#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and volume confirmation.
# Long when price breaks above Donchian upper band, price > 1d EMA50, volume > 1.8x 20-period average.
# Short when price breaks below Donchian lower band, price < 1d EMA50, volume > 1.8x 20-period average.
# Exit when price crosses back below/above Donchian mid-band.
# Uses 4h for entry timing and 1d EMA for trend direction to reduce whipsaw.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_EMA50_Volume"
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
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high)
    low_roll = pd.Series(low)
    donch_high = high_roll.rolling(window=20, min_periods=20).max().values
    donch_low = low_roll.rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Calculate 1-day EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for Donchian and EMA calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        mid = donch_mid[i]
        ema50 = ema50_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper, above 1d EMA50, volume spike
            if price > upper and price > ema50 and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower, below 1d EMA50, volume spike
            elif price < lower and price < ema50 and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Donchian mid-band
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above Donchian mid-band
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals