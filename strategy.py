#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
    # Uses Donchian breakouts for entry timing, 12h HMA for structural trend bias,
    # and volume spike for breakout validity. Designed to work in both bull and bear markets
    # by only taking breakouts in the direction of the higher timeframe trend.
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h HMA(21) - Hull Moving Average
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    wma_2x_sub = 2 * wma_half - wma_full
    
    # Pad the beginning with NaN to align lengths
    wma_2x_sub_padded = np.full(len(close_12h), np.nan)
    wma_2x_sub_padded[half_len-1:half_len-1+len(wma_2x_sub)] = wma_2x_sub
    
    hma_12h = wma(wma_2x_sub_padded, sqrt_len)
    hma_12h_padded = np.full(len(close_12h), np.nan)
    hma_12h_padded[sqrt_len-1:sqrt_len-1+len(hma_12h)] = hma_12h
    
    # Align 12h HMA to 4h timeframe
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period MA (volume spike)
        volume_filter = volume[i] > 1.5 * volume_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i-1]  # Break above prior period's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below prior period's low
        
        # Trend filter: price above/below 12h HMA
        bullish_trend = close[i] > hma_12h_aligned[i]
        bearish_trend = close[i] < hma_12h_aligned[i]
        
        # Entry conditions: breakout in direction of 12h HMA trend
        long_entry = long_breakout and bullish_trend and volume_filter
        short_entry = short_breakout and bearish_trend and volume_filter
        
        # Exit conditions: opposite breakout or trend reversal
        long_exit = short_breakout or not bullish_trend
        short_exit = long_breakout or not bearish_trend
        
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

name = "4h_12h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0