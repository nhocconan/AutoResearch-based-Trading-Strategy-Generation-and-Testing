#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe strategy using 4h trend direction (EMA200) and 1d volume filter for entry timing.
# Uses 1h Donchian breakout (20-period) with 4h EMA200 trend filter and 1d volume spike confirmation.
# Designed to work in both bull (follow 4h uptrend) and bear (follow 4h downtrend) markets.
# Entry only during active session (08-20 UTC) to reduce noise.
# Position size fixed at 0.20 to manage drawdown.
name = "1h_Donchian20_4hEMA200_1dVolFilter_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume EMA20 for spike detection
    volume_1d = df_1d['volume'].values
    vol_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_1d)
    
    # 1h Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need sufficient lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema_200_4h_aligned[i]) or 
            np.isnan(vol_ema20_1d_aligned[i]) or
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian upper + 4h EMA200 uptrend + volume spike
            if (price > donchian_upper[i] and 
                price > ema_200_4h_aligned[i] and 
                vol_current > 1.5 * vol_ema20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: Price breaks below Donchian lower + 4h EMA200 downtrend + volume spike
            elif (price < donchian_lower[i] and 
                  price < ema_200_4h_aligned[i] and 
                  vol_current > 1.5 * vol_ema20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Donchian lower or 4h trend turns down
            if (price < donchian_lower[i] or 
                price < ema_200_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Price breaks above Donchian upper or 4h trend turns up
            if (price > donchian_upper[i] or 
                price > ema_200_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals