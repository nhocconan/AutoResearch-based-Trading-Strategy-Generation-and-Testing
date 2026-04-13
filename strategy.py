#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h HMA trend filter + volume confirmation
    # Long: price > Donchian(20) high AND price > 12h HMA(21) AND volume > 1.5x 20-period average
    # Short: price < Donchian(20) low AND price < 12h HMA(21) AND volume > 1.5x 20-period average
    # Exit: opposite Donchian breakout
    # Using 4h timeframe for optimal trade frequency (target 19-50/year), 12h HMA for strong trend filter,
    # and volume confirmation to avoid false breakouts. Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    close_12h = df_12h['close'].values
    n_12h = len(close_12h)
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    
    wma_half = np.full(n_12h, np.nan)
    wma_full = np.full(n_12h, np.nan)
    hma_12h = np.full(n_12h, np.nan)
    
    for i in range(half_n, n_12h):
        wma_half[i] = wma(close_12h[i-half_n+1:i+1], half_n)
    for i in range(21, n_12h):
        wma_full[i] = wma(close_12h[i-21+1:i+1], 21)
        if i >= half_n and not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            raw_hma = 2 * wma_half[i] - wma_full[i]
            if i >= half_n + sqrt_n - 1:
                hma_12h[i] = wma(raw_hma[half_n-1:i-half_n+1+sqrt_n], sqrt_n)
    
    # Get 4h Donchian(20) for breakout with min_periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 12h HMA to 4h
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(hma_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Trend filter conditions
        bullish_trend = close[i] > hma_12h_aligned[i]  # Above HMA = bullish bias
        bearish_trend = close[i] < hma_12h_aligned[i]  # Below HMA = bearish bias
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: opposite breakout
        long_exit = short_breakout
        short_exit = long_breakout
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_12h_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0