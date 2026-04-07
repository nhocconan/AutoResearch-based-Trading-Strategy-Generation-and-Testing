#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian Channel Breakout with Daily Trend Filter and Volume Spike
# Hypothesis: Donchian(20) breakouts on 12h timeframe capture strong trends.
# In bull markets: buy breakouts above upper band when price > daily 50 EMA.
# In bear markets: sell breakdowns below lower band when price < daily 50 EMA.
# Volume filter (2x average) ensures institutional participation.
# Target: 20-50 trades/year (80-200 over 4 years) to minimize fee drag.

name = "12h_donchian20_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Daily 50 EMA for trend filter
    close_series_daily = pd.Series(df_daily['close'].values)
    ema_50_daily = close_series_daily.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_50_daily)
    
    # Calculate Donchian Channel (20-period) on 12h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_daily_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches lower band or trend turns bearish or volume drops
            if (low[i] <= donchian_lower[i] or close[i] < ema_50_daily_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price touches upper band or trend turns bullish or volume drops
            if (high[i] >= donchian_upper[i] or close[i] > ema_50_daily_aligned[i] or not vol_filter[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above upper band with volume and bullish trend
            if (high[i] > donchian_upper[i] and close[i] > ema_50_daily_aligned[i] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower band with volume and bearish trend
            elif (low[i] < donchian_lower[i] and close[i] < ema_50_daily_aligned[i] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals