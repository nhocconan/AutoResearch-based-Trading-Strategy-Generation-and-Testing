#!/usr/bin/env python3
"""
6h_WilliamsVIXFix_Breakout_1dTrend
Hypothesis: On 6h timeframe, enter long when price breaks above Donchian(20) high AND Williams VIX Fix > 0.8 (extreme fear) AND 1d trend is up (close > EMA50). Enter short when price breaks below Donchian(20) low AND Williams VIX Fix < 0.2 (extreme greed) AND 1d trend is down (close < EMA50). Uses Williams VIX Fix to detect exhaustion during extreme volatility spikes, which often precedes reversals. The Donchian breakout provides entry timing, while 1d EMA50 filters for higher-timeframe trend alignment. Designed for low trade frequency (15-30/year) with strong edge in both bull and bear markets by buying panic and selling euphoria.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams VIX Fix: measures market fear/greed using price range relative to recent high
    # VIX Fix = (Highest High in period - Low) / (Highest High in period - Lowest Low in period) * 100
    # We invert it so high values = fear, low values = greed
    lookback = 22  # ~1 month for 6h (22*6h = 5.5 days)
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    vix_fix = (highest_high - low) / np.maximum(highest_high - lowest_low, 1e-10)
    # Invert: high VIX Fix = fear (good for long), low VIX Fix = greed (good for short)
    
    # Donchian channels for breakout
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), VIX Fix (22), EMA50 (50)
    start_idx = max(donchian_period, lookback, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vix_fix[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Williams VIX Fix conditions: extreme fear/greed
        extreme_fear = vix_fix[i] > 0.8   # High fear = potential long
        extreme_greed = vix_fix[i] < 0.2  # High greed = potential short
        
        if position == 0:
            # Long: breakout up + extreme fear + 1d uptrend
            long_signal = breakout_up and extreme_fear and trend_uptrend
            
            # Short: breakout down + extreme greed + 1d downtrend
            short_signal = breakout_down and extreme_greed and trend_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR trend change OR fear subsides
            if close[i] < donchian_low[i] or not trend_uptrend or vix_fix[i] < 0.5:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR trend change OR greed subsides
            if close[i] > donchian_high[i] or not trend_downtrend or vix_fix[i] > 0.5:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WilliamsVIXFix_Breakout_1dTrend"
timeframe = "6h"
leverage = 1.0