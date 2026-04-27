#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 12-hour trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high with 12h EMA50 uptrend and volume > 1.5x average.
Short when price breaks below 20-period Donchian low with 12h EMA50 downtrend and volume > 1.5x average.
Exit when price crosses the Donchian midpoint or trend reverses.
Uses 0.25 position size to limit drawdown. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Get daily volume for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20) + EMA (50) + volume MA (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values
        price_now = close[i]
        vol_now = volume[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        midpoint = donchian_mid[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above upper band with uptrend and volume
            if price_now > upper and ema_trend > price_now * 0.99 and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band with downtrend and volume
            elif price_now < lower and ema_trend < price_now * 1.01 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price below midpoint or trend turns down
            if price_now < midpoint or ema_trend < price_now * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price above midpoint or trend turns up
            if price_now > midpoint or ema_trend > price_now * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0