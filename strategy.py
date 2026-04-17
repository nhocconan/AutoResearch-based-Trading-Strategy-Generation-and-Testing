#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data (1D)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get weekly data (1W) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Daily 20-period EMA for trend filter
    close_1d_series = pd.Series(close_1d)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly 50-period EMA for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to 1D timeframe (same as prices for 1d timeframe)
    ema20_aligned = ema20_1d  # Same timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    upper_donchian = align_htf_to_ltf(prices, df_1d, high_20)
    lower_donchian = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need Donchian, EMAs, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema20_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(upper_donchian[i]) or 
            np.isnan(lower_donchian[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: daily price above/below daily EMA20 AND weekly EMA50
        price_above_ema20 = close[i] > ema20_aligned[i]
        price_below_ema20 = close[i] < ema20_aligned[i]
        price_above_weekly_ema = close[i] > ema50_aligned[i]
        price_below_weekly_ema = close[i] < ema50_aligned[i]
        
        # Price relative to daily Donchian channels
        price_above_upper = close[i] > upper_donchian[i]
        price_below_lower = close[i] < lower_donchian[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above both EMAs
            if (price_above_upper and price_above_ema20 and price_above_weekly_ema and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume and below both EMAs
            elif (price_below_lower and price_below_ema20 and price_below_weekly_ema and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below lower Donchian OR below daily EMA20
            if (close[i] < lower_donchian[i]) or (close[i] < ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above upper Donchian OR above daily EMA20
            if (close[i] > upper_donchian[i]) or (close[i] > ema20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_EMA20_EMA50_Volume"
timeframe = "1d"
leverage = 1.0