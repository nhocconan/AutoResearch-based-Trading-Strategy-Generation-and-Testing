#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1-week EMA10 trend filter + volume confirmation.
# Long when: price breaks above Donchian upper band (20-period high) AND price > 1-week EMA10 AND volume > 1.5x 20-period average
# Short when: price breaks below Donchian lower band (20-period low) AND price < 1-week EMA10 AND volume > 1.5x 20-period average
# Exit when: price returns to Donchian midpoint (mean of upper and lower bands)
# Donchian captures breakouts, weekly EMA filters trend direction, volume confirms conviction.
# Target: 12-30 trades/year per symbol. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend).
name = "12h_Donchian20_WeeklyEMA10_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA10 on weekly data
    ema10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Donchian channels (20-period) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations (20 for Donchian + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema10_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = donchian_mid[i]
        ema10 = ema10_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above upper band, price > weekly EMA10, volume spike
            if (price > upper and price > ema10 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band, price < weekly EMA10, volume spike
            elif (price < lower and price < ema10 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint
            if price <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint
            if price >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals