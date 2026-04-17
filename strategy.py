#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 25:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend direction and volatility
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(low_1w[1:], high_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly indicators to 6h timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    # Get daily data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) channel
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(atr14_1w_aligned[i]) or 
            np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA200
        price_above_ema200 = close[i] > ema200_1w_aligned[i]
        price_below_ema200 = close[i] < ema200_1w_aligned[i]
        
        # Volatility filter: current ATR > 1.5 * weekly ATR
        # We don't have current ATR, so we use price range as proxy
        current_range = high[i] - low[i]
        volatility_filter = current_range > (1.5 * atr14_1w_aligned[i])
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Price relative to Donchian levels
        price_above_upper = close[i] > highest_20_aligned[i]
        price_below_lower = close[i] < lowest_20_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with trend, volatility, and volume
            if (price_above_upper and price_above_ema200 and volatility_filter and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with trend, volatility, and volume
            elif (price_below_lower and price_below_ema200 and volatility_filter and volume_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below Donchian lower OR trend reverses
            if (close[i] < lowest_20_aligned[i]) or (close[i] < ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above Donchian upper OR trend reverses
            if (close[i] > highest_20_aligned[i]) or (close[i] > ema200_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyTrend_Donchian20_VolVol"
timeframe = "6h"
leverage = 1.0