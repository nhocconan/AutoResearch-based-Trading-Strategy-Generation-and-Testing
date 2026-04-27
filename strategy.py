#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) with 12h EMA trend filter and volume confirmation
# In trending markets (price above/below 12h EMA), trade Donchian breakouts in trend direction.
# Volume filter ensures breakouts have conviction. Works in bull/bear via 12h trend filter.
# Target: 75-200 total trades over 4 years (~19-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Donchian channels (20-period high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA trend filter (50-period)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: volume > 1.5 x 20-period average (approx 10 hours of 4h bars)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (50), volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 12h EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + bullish 12h trend
            if price > donch_high[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low + volume + bearish 12h trend
            elif price < donch_low[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend turns bearish
            if price < donch_low[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend turns bullish
            if price > donch_high[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0