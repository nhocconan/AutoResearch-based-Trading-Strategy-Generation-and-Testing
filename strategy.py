#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21) and volume confirmation.
    # In bull/bear markets, price tends to continue in direction of weekly trend after breaking daily channels.
    # Volume confirms institutional participation. Target: 30-100 total trades over 4 years (7-25/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume MA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper = highest high of last 20 days
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian lower = lowest low of last 20 days
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for HMA21 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    if len(close_1w) >= 21:
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        # Pad the beginning with NaN to align lengths
        wma_2x_sub_padded = np.full(len(close_1w), np.nan)
        wma_2x_sub_padded[half_len-1:half_len-1+len(wma_2x_sub)] = wma_2x_sub
        hma_21 = wma(wma_2x_sub_padded[~np.isnan(wma_2x_sub_padded)], sqrt_len)
        # Pad the final HMA result to align with original array
        hma_21_final = np.full(len(close_1w), np.nan)
        start_idx = len(close_1w) - len(hma_21)
        hma_21_final[start_idx:] = hma_21
        hma_21 = hma_21_final
    else:
        hma_21 = np.full(len(close_1w), np.nan)
    
    # Align HTF indicators to 1d timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate 1d volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 20-period MA
        volume_filter = volume[i] > volume_ma[i]
        
        # Trend filter: price above/below weekly HMA21
        price_above_weekly_hma = close[i] > hma_21_aligned[i]
        price_below_weekly_hma = close[i] < hma_21_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        if volume_filter:
            # Long when price breaks above Donchian upper AND above weekly HMA
            # Short when price breaks below Donchian lower AND below weekly HMA
            long_entry = (close[i] > donch_high_aligned[i]) and price_above_weekly_hma
            short_entry = (close[i] < donch_low_aligned[i]) and price_below_weekly_hma
        
        # Exit conditions: opposite Donchian break or trend change
        long_exit = False
        short_exit = False
        
        if position == 1:
            # Exit long when price breaks below Donchian lower OR crosses below weekly HMA
            long_exit = (close[i] < donch_low_aligned[i]) or (close[i] < hma_21_aligned[i])
        elif position == -1:
            # Exit short when price breaks above Donchian upper OR crosses above weekly HMA
            short_exit = (close[i] > donch_high_aligned[i]) or (close[i] > hma_21_aligned[i])
        
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

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0