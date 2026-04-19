#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
# Long when: price breaks above Donchian high(20) + weekly EMA20 uptrend + volume > 1.5x 20-day average
# Short when: price breaks below Donchian low(20) + weekly EMA20 downtrend + volume > 1.5x 20-day average
# Exit when: price crosses opposite Donchian band (long exits at low, short exits at high)
# Weekly EMA20 filters trend direction to avoid counter-trend trades. Volume confirms breakout strength.
# Target: 15-25 trades/year per symbol. Works in bull (buy breakouts) and bear (sell breakdowns).
name = "1d_Donchian20_WeeklyEMA20_Volume"
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
    
    # Weekly data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on weekly data
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 20-day volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        ema20 = ema20_1w_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above upper band + weekly uptrend + volume spike
            if (price > upper and ema20 > ema20_1w_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower band + weekly downtrend + volume spike
            elif (price < lower and ema20 < ema20_1w_aligned[i-1] and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals