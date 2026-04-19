#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) upper band, price > EMA34(12h), volume > 1.5x 20-period average
# Short when: price breaks below Donchian(20) lower band, price < EMA34(12h), volume > 1.5x 20-period average
# Exit when: price crosses back through the Donchian middle line (mean of upper/lower bands)
# Donchian channels provide clear breakout signals, EMA34 filters trend direction, volume confirms conviction.
# Target: 20-40 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "4h_Donchian_EMA34_VolumeBreakout"
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
    
    # 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h data
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema34_12h_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper, price > EMA34, volume spike
            if (price > upper and price > ema34 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower, price < EMA34, volume spike
            elif (price < lower and price < ema34 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below Donchian middle
            if price < middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above Donchian middle
            if price > middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals