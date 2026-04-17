#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
# Uses price channels (Donchian) for entries, weekly EMA for trend filter, volume for confirmation.
# Designed to work in bull (breakouts with trend) and bear (mean reversion via Donchian reversion).
# Target: 10-30 trades/year to avoid fee drag on daily timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily Donchian(20) channels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need weekly EMA50 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d[i]) or 
            np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA50
        price_above_ema = close[i] > ema50_1d[i]
        price_below_ema = close[i] < ema50_1d[i]
        
        # Price relative to daily Donchian channels
        price_above_upper = close[i] > high_max[i]
        price_below_lower = close[i] < low_min[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with volume and above weekly EMA50
            if (price_above_upper and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with volume and below weekly EMA50
            elif (price_below_lower and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower OR below weekly EMA50
            if (close[i] < low_min[i]) or (close[i] < ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper OR above weekly EMA50
            if (close[i] > high_max[i]) or (close[i] > ema50_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA50_Volume"
timeframe = "1d"
leverage = 1.0