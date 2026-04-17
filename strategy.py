#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR and EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily ATR(14) moving average for volatility regime filter
    atr_ma50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align daily indicators to 12h timeframe
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    ema50_12h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    atr_ma50_12h = align_htf_to_ltf(prices, df_1d, atr_ma50)
    
    # Calculate 12h Donchian channel (20-period)
    high_ma20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # Need daily ATR(14) MA50, EMA50, Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(atr_14_12h[i]) or 
            np.isnan(ema50_12h[i]) or 
            np.isnan(atr_ma50_12h[i]) or 
            np.isnan(high_ma20[i]) or 
            np.isnan(low_ma20[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR > 1.2 * ATR MA (high volatility regime)
        vol_filter = atr_14_12h[i] > (1.2 * atr_ma50_12h[i])
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema50_12h[i]
        price_below_ema = close[i] < ema50_12h[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > high_ma20[i]  # Break above upper band
        breakout_down = low[i] < low_ma20[i]  # Break below lower band
        
        # Volume confirmation
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        if position == 0:
            # Long: Break above upper Donchian band with volume, above EMA50, and high volatility
            if (breakout_up and volume_filter and price_above_ema and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with volume, below EMA50, and high volatility
            elif (breakout_down and volume_filter and price_below_ema and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below EMA50 OR ATR drops below normal (volatility collapse)
            if (close[i] < ema50_12h[i]) or (atr_14_12h[i] < (0.8 * atr_ma50_12h[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above EMA50 OR ATR drops below normal (volatility collapse)
            if (close[i] > ema50_12h[i]) or (atr_14_12h[i] < (0.8 * atr_ma50_12h[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_EMA50_Vol_VolatilityFilter"
timeframe = "12h"
leverage = 1.0