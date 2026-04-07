#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Daily Donchian Breakout with Volume Confirmation
# Hypothesis: Donchian(20) breakouts on 12h chart, confirmed by volume and daily trend,
# capture institutional moves. Works in bull (breakouts continue) and bear (breakouts fail = mean reversion).
# Uses volume > 1.5x 20-period average to confirm institutional participation.
# Target: 15-25 trades/year (60-100 over 4 years) to avoid fee drag.

name = "12h_donchian20_volume_trend_v1"
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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    
    # Donchian(20) on 12h
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
            np.isnan(vol_ma[i]) or np.isnan(daily_ema50[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned daily trend (use previous day's EMA to avoid look-ahead)
        daily_idx = i // 2  # 2x 12h bars per day
        if daily_idx < 1:
            signals[i] = 0.0
            continue
        daily_trend = daily_ema50[daily_idx-1]  # Previous day's EMA
        
        if position == 1:  # Long position
            # Exit: price falls to Donchian low or trend turns bearish
            if close[i] <= donchian_low[i] or close[i] < daily_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: price rises to Donchian high or trend turns bullish
            if close[i] >= donchian_high[i] or close[i] > daily_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with volume and bullish trend
            if high[i] > donchian_high[i] and close[i] > donchian_high[i] and vol_filter[i] and close[i] > daily_trend:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with volume and bearish trend
            elif low[i] < donchian_low[i] and close[i] < donchian_low[i] and vol_filter[i] and close[i] < daily_trend:
                position = -1
                signals[i] = -0.25
    
    return signals