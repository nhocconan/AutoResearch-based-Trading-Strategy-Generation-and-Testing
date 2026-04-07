#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Breakout with Daily Filter and Volume Confirmation
# Hypothesis: On the 12-hour timeframe, price breaking above/below 20-period Donchian channels 
# with daily trend alignment and volume confirmation captures institutional moves. 
# Daily trend filter ensures we trade with the higher timeframe momentum, reducing whipsaws. 
# Volume confirmation ensures only significant breaks trigger entries. 
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue) 
# because we follow the daily trend direction. Tight entry conditions limit trades to 12-37/year.

name = "12h_donchian20_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (use previous day's data)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    daily_close_series = pd.Series(daily_close)
    daily_ema50 = daily_close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA50 to 12h timeframe (use previous day's EMA)
    daily_ema50_shifted = np.roll(daily_ema50, 1)
    if len(daily_ema50_shifted) > 1:
        daily_ema50_shifted[0] = daily_ema50_shifted[1]
    daily_ema50_aligned = align_htf_to_ltf(prices, df_daily, daily_ema50_shifted)
    
    # Calculate 20-period Donchian channels on 12h high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls to Donchian low or trend changes or volume drops
            if (close[i] <= donchian_low[i] or 
                close[i] < daily_ema50_aligned[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to Donchian high or trend changes or volume drops
            if (close[i] >= donchian_high[i] or 
                close[i] > daily_ema50_aligned[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with daily uptrend and volume
            if (high[i] > donchian_high[i] and 
                close[i] > donchian_high[i] and 
                close[i] > daily_ema50_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with daily downtrend and volume
            elif (low[i] < donchian_low[i] and 
                  close[i] < donchian_low[i] and 
                  close[i] < daily_ema50_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals