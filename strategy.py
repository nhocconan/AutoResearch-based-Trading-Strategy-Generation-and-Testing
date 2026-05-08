#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with daily trend filter and volume spike
# Long when price breaks above upper band, daily EMA(34) uptrend, and volume spike
# Short when price breaks below lower band, daily EMA(34) downtrend, and volume spike
# Donchian bands from prior day provide structured support/resistance
# Daily EMA filters for higher timeframe trend alignment
# Volume spike confirms institutional participation; avoids false breakouts
# Targets 50-150 total trades over 4 years (12-37/year) to balance opportunity and cost

name = "12h_Donchian20_DailyTrend_Volume"
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
    
    # Get daily data once for Donchian bands and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) bands from previous day
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Upper band: highest high of last 20 days (excluding today)
    upper_band = pd.Series(daily_high).rolling(window=20, min_periods=20).max().shift(1).values
    # Lower band: lowest low of last 20 days (excluding today)
    lower_band = pd.Series(daily_low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian bands to 12h timeframe (available after daily close)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = upper_band_aligned[i]
        lower = lower_band_aligned[i]
        ema34_1d_val = ema34_1d_aligned[i]
        price = close[i]
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