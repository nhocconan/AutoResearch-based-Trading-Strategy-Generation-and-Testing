#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d trend filter and volume confirmation
# Williams Fractals identify key support/resistance levels; breakouts with volume
# and higher timeframe trend capture momentum while minimizing false breaks.
# Works in bull/bear by filtering breakout direction with 1d EMA trend.
# Target: 75-200 total trades over 4 years (~19-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractal calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams Fractals for each 1d bar
    # Bearish fractal: high[n-2] is the highest of high[n-4:n-1] and high[n:n+3]
    # Bullish fractal: low[n-2] is the lowest of low[n-4:n-1] and low[n:n+3]
    bearish_fractal = np.full(len(df_1d), np.nan)
    bullish_fractal = np.full(len(df_1d), np.nan)
    
    for i in range(2, len(df_1d) - 2):
        # Bearish fractal: high at i is the highest in window [i-2:i+3]
        window_high = high_1d[i-2:i+3]
        if high_1d[i] == np.max(window_high):
            bearish_fractal[i] = high_1d[i]
        
        # Bullish fractal: low at i is the lowest in window [i-2:i+3]
        window_low = low_1d[i-2:i+3]
        if low_1d[i] == np.min(window_low):
            bullish_fractal[i] = low_1d[i]
    
    # Williams Fractals need 2 extra 1d bars after the center bar for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # 1d EMA trend filter (34-period)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: volume > 1.5 x 24-period average (4 days of 4h bars)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(23, n):
        vol_ma_24[i] = np.mean(volume[i-23:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (5 bars for fractal), EMA (34), volume MA (24)
    start_idx = max(5, 34, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_24[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_34_aligned[i]
        bearish_trend = price < ema_34_aligned[i]
        
        if position == 0:
            # Long: break above bullish fractal with volume and bullish trend
            if price > bullish_fractal_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below bearish fractal with volume and bearish trend
            elif price < bearish_fractal_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to bearish fractal (mean reversion) or trend turns bearish
            if price <= bearish_fractal_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to bullish fractal (mean reversion) or trend turns bullish
            if price >= bullish_fractal_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Williams_Fractal_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0