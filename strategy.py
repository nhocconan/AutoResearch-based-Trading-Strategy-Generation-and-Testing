#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d trend filter (EMA50).
# Long when price breaks above Donchian upper band (20-period high) with volume > 1.5x EMA20 and price > 1d EMA50.
# Short when price breaks below Donchian lower band (20-period low) with volume > 1.5x EMA20 and price < 1d EMA50.
# Uses discrete position sizing (0.25) to minimize churn. Designed for 4h timeframe to balance trade frequency and win rate.
# Works in both bull and bear markets by following 1d EMA50 trend filter.
name = "4h_Donchian20_1dVolume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d volume EMA(20) for confirmation
    vol_ema20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20)
    
    # 1d EMA(50) trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian to be valid
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ema20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_confirm = volume[i] > (1.5 * vol_ema20_aligned[i])
        
        if position == 0:
            # Long: price > Donchian upper + volume confirmation + price > 1d EMA50
            if price > highest_high[i] and vol_confirm and price > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower + volume confirmation + price < 1d EMA50
            elif price < lowest_low[i] and vol_confirm and price < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Donchian lower (breakdown) or loss of trend
            if price < lowest_low[i] or price < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Donchian upper (breakout) or loss of trend
            if price > highest_high[i] or price > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals