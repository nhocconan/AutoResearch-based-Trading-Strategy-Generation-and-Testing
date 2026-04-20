#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h VWAP trend filter + volume confirmation
# - Long when price breaks above Donchian high (20) AND price > 12h VWAP AND volume > 1.5x average
# - Short when price breaks below Donchian low (20) AND price < 12h VWAP AND volume > 1.5x average
# - Exit when price crosses back through the Donchian midpoint or volume dries up
# - Donchian provides clear breakout levels, VWAP filters for institutional trend, volume confirms conviction
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 15-40 trades per year per symbol (60-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for VWAP calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate VWAP on 12h timeframe
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    vwap_num = (typical_price_12h * volume_12h).cumsum()
    vwap_den = volume_12h.cumsum()
    vwap_12h = vwap_num / vwap_den
    # Handle division by zero at start
    vwap_12h = np.where(vwap_den != 0, vwap_12h, typical_price_12h)
    
    # Align 12h VWAP to 4h timeframe
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    highest_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2
    
    # Calculate average volume for confirmation
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if NaN in indicators
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(vwap_12h_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_4h[i]
        vwap = vwap_12h_aligned[i]
        volume = volume_4h[i]
        avg_vol = avg_volume_20[i]
        upper_channel = highest_high_20[i]
        lower_channel = lowest_low_20[i]
        mid_channel = donchian_mid[i]
        
        if position == 0:
            # Long entry: price breaks above upper Donchian AND price > VWAP AND volume > 1.5x average
            if price > upper_channel and price > vwap and volume > 1.5 * avg_vol:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian AND price < VWAP AND volume > 1.5x average
            elif price < lower_channel and price < vwap and volume > 1.5 * avg_vol:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint OR volume drops below average
            if price < mid_channel or volume < avg_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint OR volume drops below average
            if price > mid_channel or volume < avg_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_VWAP_VolumeFilter"
timeframe = "4h"
leverage = 1.0