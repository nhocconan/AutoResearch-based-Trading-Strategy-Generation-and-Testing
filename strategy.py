#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter (HMA21) + volume confirmation
# In bull/bear markets: Donchian breakouts capture strong moves, weekly HMA filter avoids counter-trend trades
# Volume confirmation ensures breakout validity. Position size 0.25 to limit drawdown.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in both bull/bear: trend filter adapts to weekly market direction.

name = "1d_1w_donchian_trend_vol_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    hma_21 = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(data, window):
            if len(data) < window:
                return np.full_like(data, np.nan)
            weights = np.arange(1, window + 1)
            wma_vals = np.full(len(data), np.nan)
            for i in range(window - 1, len(data)):
                wma_vals[i] = np.dot(data[i - window + 1:i + 1], weights) / weights.sum()
            return wma_vals
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        raw_hma = 2 * wma_half - wma_full
        hma_21 = wma(raw_hma, sqrt_len)
    
    # Align 1w HMA to 1d timeframe
    hma_21_1d = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1d[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * average volume
        vol_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions: price closes below Donchian low OR trend turns bearish
            if close[i] < donchian_low[i] or close[i] < hma_21_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian high OR trend turns bullish
            if close[i] > donchian_high[i] or close[i] > hma_21_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout + volume confirmation + trend filter
            if vol_confirmed:
                # Long: price breaks above Donchian high in bullish weekly trend
                if high[i] > donchian_high[i] and close[i] > hma_21_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low in bearish weekly trend
                elif low[i] < donchian_low[i] and close[i] < hma_21_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals