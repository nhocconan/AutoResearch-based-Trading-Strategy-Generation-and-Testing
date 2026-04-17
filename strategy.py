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
    
    # Get daily data for ATR and EMA trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_close[0] = high_low[0]  # first value
    low_close[0] = high_low[0]   # first value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_4h = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate daily EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # Need daily EMA50, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_4h[i]) or 
            np.isnan(atr14_4h[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * price (avoid low volatility chop)
        volatility_filter = atr14_4h[i] > (0.005 * close[i])
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema50_4h[i]
        price_below_ema = close[i] < ema50_4h[i]
        
        if position == 0:
            # Long: Price above EMA50 with volume and volatility
            if (price_above_ema and volume_filter and volatility_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price below EMA50 with volume and volatility
            elif (price_below_ema and volume_filter and volatility_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below EMA50
            if close[i] < ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above EMA50
            if close[i] > ema50_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA50_ATR_Volume_Filter"
timeframe = "4h"
leverage = 1.0