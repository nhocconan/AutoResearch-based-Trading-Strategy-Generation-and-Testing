#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout with Daily Trend Filter
# Hypothesis: Donchian(20) breakouts on 4h timeframe capture momentum. 
# Daily trend filter (price > EMA50) ensures we only trade in the direction of higher timeframe trend.
# Works in bull markets: breakouts above upper band continue up. 
# Works in bear markets: breakouts below lower band continue down. 
# Volume filter ensures breakouts have institutional participation.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_donchian20_daily_trend_volume_v1"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    daily_close = df_daily['close'].values
    daily_close_series = pd.Series(daily_close)
    daily_ema50 = daily_close_series.ewm(span=50, adjust=False).mean().values
    
    # Align daily EMA50 to 4h timeframe (use previous day's EMA)
    prev_daily_ema50 = np.roll(daily_ema50, 1)
    prev_daily_ema50[0] = prev_daily_ema50[1] if len(prev_daily_ema50) > 1 else 0
    daily_ema50_aligned = align_htf_to_ltf(prices, df_daily, prev_daily_ema50)
    
    # Calculate Donchian channels on 4h (20-period high/low)
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
            # Exit: price falls below Donchian low or trend changes or volume drops
            if (close[i] <= donchian_low[i] or 
                close[i] < daily_ema50_aligned[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high or trend changes or volume drops
            if (close[i] >= donchian_high[i] or 
                close[i] > daily_ema50_aligned[i] or 
                not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and uptrend
            if (high[i] > donchian_high[i] and 
                close[i] > donchian_high[i] and 
                close[i] > daily_ema50_aligned[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and downtrend
            elif (low[i] < donchian_low[i] and 
                  close[i] < donchian_low[i] and 
                  close[i] < daily_ema50_aligned[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals