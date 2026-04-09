#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 12h Supertrend(ATR=10, mult=3.0) trend filter + volume confirmation (1.5x 20-period avg)
# Donchian breakouts capture strong momentum moves; Supertrend ensures we trade with higher timeframe trend direction
# Volume confirmation filters weak breakouts. Supertrend avoids counter-trend whipsaws in ranging markets.
# Works in bull/bear: Supertrend adapts to volatility and trend changes.
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_12h_donchian_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3.0)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10)
    atr_10 = np.full(len(close_12h), np.nan)
    for i in range(len(close_12h)):
        if i < 10:
            atr_10[i] = np.nan
        else:
            atr_10[i] = np.nanmean(tr[i-9:i+1])
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + (3.0 * atr_10)
    lowerband = hl2 - (3.0 * atr_10)
    
    supertrend = np.full(len(close_12h), np.nan)
    direction = np.full(len(close_12h), np.nan)  # 1 = uptrend, -1 = downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr_10[i]) or np.isnan(close_12h[i-1]):
            supertrend[i] = np.nan
            direction[i] = np.nan
        else:
            if close_12h[i-1] > supertrend[i-1]:
                upperband[i] = min(upperband[i], upperband[i-1])
            else:
                lowerband[i] = max(lowerband[i], lowerband[i-1])
            
            if close_12h[i] > upperband[i]:
                direction[i] = 1
            elif close_12h[i] < lowerband[i]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
                if direction[i] == 1 and lowerband[i] < lowerband[i-1]:
                    lowerband[i] = lowerband[i-1]
                if direction[i] == -1 and upperband[i] > upperband[i-1]:
                    upperband[i] = upperband[i-1]
            
            supertrend[i] = lowerband[i] if direction[i] == 1 else upperband[i]
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR Supertrend turns bearish
            if close[i] < donchian_low[i] or direction_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR Supertrend turns bullish
            if close[i] > donchian_high[i] or direction_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + Supertrend trend filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND Supertrend bullish (uptrend)
                if close[i] > donchian_high[i] and direction_aligned[i] == 1:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND Supertrend bearish (downtrend)
                elif close[i] < donchian_low[i] and direction_aligned[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals