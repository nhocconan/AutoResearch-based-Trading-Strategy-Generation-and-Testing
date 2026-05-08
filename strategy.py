#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Fractal breakout with 1d volume confirmation and 1w trend filter.
# Long when bullish fractal breaks above resistance AND 1d volume > 1.5x 20-period average AND 1w EMA50 uptrend.
# Short when bearish fractal breaks below support AND 1d volume > 1.5x 20-period average AND 1w EMA50 downtrend.
# Exit when price crosses back inside the fractal structure (between support and resistance).
# Uses 4h timeframe with 1d volume and 1w trend for higher timeframe context.
# Target: 75-200 total trades over 4 years (19-50/year) with controlled frequency to avoid fee drag.

name = "4h_Williams_Fractal_Breakout_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for volume filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # 1d volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(df_d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_filter_d = align_htf_to_ltf(prices, df_d, df_d['volume'].values > (1.5 * vol_ma20))
    
    # 1w EMA50 trend filter
    ema50_w = pd.Series(df_w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    trend_up = ema50_w_aligned > df_w['close'].shift(1).values  # Using prior close for trend direction
    trend_up_aligned = align_htf_to_ltf(prices, df_w, trend_up)
    trend_down = ema50_w_aligned < df_w['close'].shift(1).values
    trend_down_aligned = align_htf_to_ltf(prices, df_w, trend_down)
    
    # Calculate Williams Fractals on 4h data
    def williams_fractals(high, low, n=2):
        """Calculate Williams Fractals: bearish (high) and bullish (low)"""
        n = int(n)
        if n < 1:
            return np.zeros_like(high, dtype=bool), np.zeros_like(high, dtype=bool)
        
        bearish = np.zeros_like(high, dtype=bool)
        bullish = np.zeros_like(high, dtype=bool)
        
        for i in range(n, len(high) - n):
            # Bearish fractal: high is highest in window
            if high[i] == np.max(high[i-n:i+n+1]):
                bearish[i] = True
            # Bullish fractal: low is lowest in window
            if low[i] == np.min(low[i-n:i+n+1]):
                bullish[i] = True
        
        return bearish, bullish
    
    bearish_fractal, bullish_fractal = williams_fractals(high, low, 2)
    
    # Fractal levels for support/resistance
    # For bullish fractal, use the low as support level
    # For bearish fractal, use the high as resistance level
    bullish_level = np.where(bullish_fractal, low, np.nan)
    bearish_level = np.where(bearish_fractal, high, np.nan)
    
    # Forward fill to get the most recent fractal level
    bullish_level_series = pd.Series(bullish_level)
    bullish_level_ff = bullish_level_series.ffill().values
    
    bearish_level_series = pd.Series(bearish_level)
    bearish_level_ff = bearish_level_series.ffill().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bullish_level_ff[i]) or np.isnan(bearish_level_ff[i]) or 
            np.isnan(volume_filter_d[i]) or np.isnan(trend_up_aligned[i]) or 
            np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above most recent bearish fractal (resistance)
            # AND volume filter AND 1w uptrend
            long_cond = (close[i] > bearish_level_ff[i]) and volume_filter_d[i] and trend_up_aligned[i]
            # Short conditions: price breaks below most recent bullish fractal (support)
            # AND volume filter AND 1w downtrend
            short_cond = (close[i] < bullish_level_ff[i]) and volume_filter_d[i] and trend_down_aligned[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below most recent bullish fractal (support)
            if close[i] < bullish_level_ff[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above most recent bearish fractal (resistance)
            if close[i] > bearish_level_ff[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals