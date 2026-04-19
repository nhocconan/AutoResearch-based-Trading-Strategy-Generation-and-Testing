#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h timeframe with weekly Donchian breakout + weekly trend filter + volume confirmation
# Uses weekly price channels for trend direction and breakout signals, avoiding noise of lower timeframes
# Weekly trend filter ensures alignment with higher timeframe momentum, reducing whipsaws
# Volume confirmation ensures breakouts have participation, improving reliability in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined risk control

name = "6h_1wDonchian20_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    # Donchian Upper = highest high over past 20 weekly periods
    # Donchian Lower = lowest low over past 20 weekly periods
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly trend filter: price above/below 50-week EMA
    # Using 50-week EMA as trend filter to ensure we only trade in direction of higher timeframe trend
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average (on 6t timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 periods for Donchian calculation
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        highest_20 = highest_20_aligned[i]
        lowest_20 = lowest_20_aligned[i]
        ema_50 = ema_50_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: Price breaks above weekly Donchian upper band AND price above weekly EMA50 (uptrend) AND volume confirmation
            if price > highest_20 and price > ema_50 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian lower band AND price below weekly EMA50 (downtrend) AND volume confirmation
            elif price < lowest_20 and price < ema_50 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price closes below weekly Donchian lower band OR price below weekly EMA50 (trend change)
            if price < lowest_20 or price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price closes above weekly Donchian upper band OR price above weekly EMA50 (trend change)
            if price > highest_20 or price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals