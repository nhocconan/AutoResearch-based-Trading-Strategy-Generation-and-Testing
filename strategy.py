#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-period high, weekly EMA(20) uptrend, and volume spike
# Short when price breaks below 20-period low, weekly EMA(20) downtrend, and volume spike
# Weekly EMA ensures alignment with longer-term trend to avoid counter-trend trades
# Volume spike confirms institutional participation, reducing false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_Donchian20_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(20) for trend filter
    weekly_close = df_1w['close'].values
    ema20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian channels (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema20_1w_val = ema20_1w_aligned[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, weekly uptrend, volume spike
            if price > upper and price > ema20_1w_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, weekly downtrend, volume spike
            elif price < lower and price < ema20_1w_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low or weekly trend turns down
            if price < lower or price < ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above Donchian high or weekly trend turns up
            if price > upper or price > ema20_1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals