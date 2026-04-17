#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for price action
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (20-period)
    high_roll_max = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_1d = align_htf_to_ltf(prices, df_1d, high_roll_max)
    donchian_low_1d = align_htf_to_ltf(prices, df_1d, low_roll_min)
    
    # Volume filter: current volume > 1.3 * 20-day average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 34  # Need weekly EMA34, daily Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1d[i]) or 
            np.isnan(donchian_high_1d[i]) or 
            np.isnan(donchian_low_1d[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.3 * volume_ma20[i])
        
        # Trend filter: price above/below weekly EMA34
        price_above_ema = close[i] > ema34_1d[i]
        price_below_ema = close[i] < ema34_1d[i]
        
        # Price relative to daily Donchian channels
        price_above_donchian_high = close[i] > donchian_high_1d[i]
        price_below_donchian_low = close[i] < donchian_low_1d[i]
        
        if position == 0:
            # Long: Price breaks above daily Donchian high with volume and above weekly EMA34
            if (price_above_donchian_high and price_above_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian low with volume and below weekly EMA34
            elif (price_below_donchian_low and price_below_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below daily Donchian low OR below weekly EMA34
            if (close[i] < donchian_low_1d[i]) or (close[i] < ema34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above daily Donchian high OR above weekly EMA34
            if (close[i] > donchian_high_1d[i]) or (close[i] > ema34_1d[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA34_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0