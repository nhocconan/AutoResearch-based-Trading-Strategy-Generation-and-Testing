#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume confirmation and 1w EMA40 trend filter.
# Long when: price breaks above 20-day high with volume > 1.8x 50-day average and price > 1w EMA40
# Short when: price breaks below 20-day low with volume > 1.8x 50-day average and price < 1w EMA40
# Exit when price returns to the 20-day midpoint or reverses to opposite side of the channel.
# Uses daily price channels for trend following, volume to confirm breakouts,
# and weekly trend to avoid counter-trend trades. Designed for ~10-20 trades/year per symbol.
name = "1d_Donchian20_Volume_EMA40Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 20-period Donchian channels (using daily data)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2
    
    # 1w EMA40 for trend filter
    ema_40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_40_1w)
    
    # Volume average (50-period) for confirmation
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_40_1w_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_50[i]
        
        # Get current levels
        high_20_val = high_20[i]
        low_20_val = low_20[i]
        mid_20_val = mid_20[i]
        ema_40 = ema_40_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price > 20-day high with volume confirmation and uptrend
            if price > high_20_val and vol > 1.8 * vol_ma and price > ema_40:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < 20-day low with volume confirmation and downtrend
            elif price < low_20_val and vol > 1.8 * vol_ma and price < ema_40:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to 20-day midpoint or breaks below 20-day low
            if price <= mid_20_val or price < low_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 20-day midpoint or breaks above 20-day high
            if price >= mid_20_val or price > high_20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals