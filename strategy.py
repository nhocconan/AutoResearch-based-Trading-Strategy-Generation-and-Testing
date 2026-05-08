#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily trend filter and volume spike
# Long when price breaks above 20-period high, daily EMA(34) uptrend, and volume spike
# Short when price breaks below 20-period low, daily EMA(34) downtrend, and volume spike
# Donchian channels provide structural support/resistance
# Daily EMA filters for higher timeframe trend alignment
# Volume spike confirms institutional participation; avoids false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_Donchian20_1dTrend_Volume"
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
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    daily_close = df_1d['close'].values
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian(20) on 12h data
    # Upper band = 20-period high, Lower band = 20-period low
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
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: price breaks above upper band, daily uptrend, volume spike
            if price > upper and price > ema34_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, daily downtrend, volume spike
            elif price < lower and price < ema34_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower band or daily trend turns down
            if price < lower or price < ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper band or daily trend turns up
            if price > upper or price > ema34_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals