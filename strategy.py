#!/usr/bin/env python3
"""
6h 48-hour Donchian Breakout with 12h Volatility Filter and 12h Trend Confirmation
Breakout above/below 48-hour high/low with volatility expansion and trend alignment
Designed for low-frequency, high-conviction trades in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for volatility and trend filters
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(10) for volatility measurement
    high_low = high_12h - low_12h
    high_close = np.abs(high_12h - np.roll(close_12h, 1))
    low_close = np.abs(low_12h - np.roll(close_12h, 1))
    high_close[0] = high_low[0]  # first value
    low_close[0] = high_low[0]   # first value
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_10_aligned = align_htf_to_ltf(prices, df_12h, atr_10)
    
    # Calculate 12h EMA21 for trend filter
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 48-hour Donchian channels (2 periods of 12h)
    # Since we're on 6h timeframe, 48h = 8 periods
    donchian_period = 8
    high_max = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_min = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(atr_10_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or
            np.isnan(high_max[i]) or np.isnan(low_min[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr_val = atr_10_aligned[i]
        ema_trend = ema_21_12h_aligned[i]
        donchian_high = high_max[i]
        donchian_low = low_min[i]
        
        # Volatility expansion: current ATR > 1.2 * average ATR
        # (using the aligned ATR as proxy for recent volatility)
        vol_expansion = atr_val > (1.2 * np.nanmedian(atr_10_aligned[max(0, i-24):i+1]))
        
        if position == 0:
            # Long: breakout above 48h high + volatility expansion + above 12h EMA21
            if (price > donchian_high and 
                vol_expansion and 
                price > ema_trend):
                signals[i] = 0.25
                position = 1
            # Short: breakout below 48h low + volatility expansion + below 12h EMA21
            elif (price < donchian_low and 
                  vol_expansion and 
                  price < ema_trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below 48h low or trend reversal
            if price < donchian_low or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 48h high or trend reversal
            if price > donchian_high or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_48hDonchian_VolExpansion_12hTrend"
timeframe = "6h"
leverage = 1.0