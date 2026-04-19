#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with weekly trend filter and volume confirmation.
# Long when: Price breaks above 20-period Donchian high, weekly EMA50 upward, volume > 1.8x 20-period average
# Short when: Price breaks below 20-period Donchian low, weekly EMA50 downward, volume > 1.8x 20-period average
# Exit when: Price crosses back through the 20-period Donchian midpoint
# Donchian channels capture breakout momentum, weekly EMA filters trend direction, volume confirms strength.
# Target: 25-35 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "4h_Donchian20_WeeklyEMA50_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to 4H timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 20-period Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for weekly EMA50 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        d_mid = donchian_mid[i]
        ema50 = ema50_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high, weekly EMA50 upward, volume spike
            if (high_price > d_high and close[i-1] <= d_high and 
                ema50 > ema50_1w_aligned[i-1] and vol > 1.8 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below Donchian low, weekly EMA50 downward, volume spike
            elif (low_price < d_low and close[i-1] >= d_low and 
                  ema50 < ema50_1w_aligned[i-1] and vol > 1.8 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below Donchian midpoint
            if price < d_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above Donchian midpoint
            if price > d_mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals