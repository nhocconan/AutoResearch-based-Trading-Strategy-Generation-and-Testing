#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour trend filter and volume confirmation.
In bull markets: long on upper band breakout with 12h uptrend.
In bear markets: short on lower band breakout with 12h downtrend.
Volume confirms breakout strength. Target: 25-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h close for trend comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + EMA (50) + volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        vol_avg = vol_avg_20[i]
        
        # Donchian levels
        upper_band = high_20[i]
        lower_band = low_20[i]
        
        # Trend filter: 12h price vs EMA
        price_12h = close_12h_aligned[i]
        ema_12h = ema_50_12h_aligned[i]
        is_uptrend = price_12h > ema_12h
        is_downtrend = price_12h < ema_12h
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: breakout above upper band with uptrend and volume
            if price_now > upper_band and is_uptrend and vol_filter:
                signals[i] = size
                position = 1
            # Short: breakout below lower band with downtrend and volume
            elif price_now < lower_band and is_downtrend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below midpoint or trend changes
            midpoint = (upper_band + lower_band) / 2
            if price_now < midpoint or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above midpoint or trend changes
            midpoint = (upper_band + lower_band) / 2
            if price_now > midpoint or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0