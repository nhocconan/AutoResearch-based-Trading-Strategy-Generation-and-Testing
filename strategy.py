#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with 1-week EMA20 trend filter and volume confirmation
# Long when close > 20-day high AND close > 1-week EMA20 AND volume > 1.5x 20-day average volume
# Short when close < 20-day low AND close < 1-week EMA20 AND volume > 1.5x 20-day average volume
# Exit when price crosses back inside the Donchian channel (opposite side)
# Uses Donchian channels for volatility breakouts, weekly EMA for trend alignment, volume for confirmation
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and cost
# Works in bull (breakouts) and bear (mean reversion via trend filter) markets

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data ONCE before loop for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian Channel on 1d (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1-week EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (20 for Donchian + buffer)
    start = 25
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: close above Donchian high + above 1w EMA20 + volume confirmation
            if (price > donchian_high[i] and price > ema20_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: close below Donchian low + below 1w EMA20 + volume confirmation
            elif (price < donchian_low[i] and price < ema20_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes back inside Donchian channel (below Donchian low)
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes back inside Donchian channel (above Donchian high)
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_1wEMA20_Volume"
timeframe = "1d"
leverage = 1.0