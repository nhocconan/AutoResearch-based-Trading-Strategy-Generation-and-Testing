#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + volume confirmation + 1d chop regime filter
# In choppy markets (1d CHOP > 61.8): mean reversion at Donchian bands
# In trending markets (1d CHOP < 38.2): breakout continuation
# Uses 12h for entries/exits, 1d for regime detection
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts to volatility regime via chop filter

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr_1d[i] = max(tr0, tr1, tr2)
    
    # ATR(14) - sum of TR
    atr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d[i] = np.nan
        elif i == 14:
            atr_1d[i] = np.nansum(tr_1d[:15])
        else:
            atr_1d[i] = atr_1d[i-1] - (atr_1d[i-1] / 14) + tr_1d[i]
    
    # Max/min high/low over 14 periods
    max_high_1d = np.full(len(df_1d), np.nan)
    min_low_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i < 14:
            max_high_1d[i] = np.nan
            min_low_1d[i] = np.nan
        else:
            max_high_1d[i] = np.nanmax(high_1d[i-13:i+1])
            min_low_1d[i] = np.nanmin(low_1d[i-13:i+1])
    
    # Chopiness Index
    chop_1d = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if atr_1d[i] > 0 and (max_high_1d[i] - min_low_1d[i]) > 0:
            log_sum = np.log10(atr_1d[i] * 14) - np.log10(max_high_1d[i] - min_low_1d[i])
            chop_1d[i] = 100 * log_sum / np.log10(14)
        else:
            chop_1d[i] = 50.0  # neutral
    
    # Align 1d chop to 12h
    chop_12h = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.nanmax(high[i-20:i+1])
            donchian_low[i] = np.nanmin(low[i-20:i+1])
    
    # Volume confirmation: 12h volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.nanmean(volume[i-20:i+1])
    volume_ok = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(chop_12h[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_12h[i]
        vol_confirm = volume_ok[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if chop > 61.8:  # Choppy regime - mean reversion
                # Exit when price reaches middle of channel
                mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] >= mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Trending regime - breakout continuation
                # Exit when price closes below Donchian low
                if close[i] <= donchian_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:  # Short position
            # Exit conditions
            if chop > 61.8:  # Choppy regime - mean reversion
                # Exit when price reaches middle of channel
                mid = (donchian_high[i] + donchian_low[i]) / 2
                if close[i] <= mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:  # Trending regime - breakout continuation
                # Exit when price closes above Donchian high
                if close[i] >= donchian_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime and volume
            if vol_confirm:  # Only trade with volume confirmation
                if chop > 61.8:  # Choppy regime - mean reversion
                    # Go long at Donchian low, short at Donchian high
                    if close[i] <= donchian_low[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] >= donchian_high[i]:
                        position = -1
                        signals[i] = -0.25
                else:  # Trending regime - breakout continuation
                    # Go long on breakout above, short on breakdown below
                    if close[i] > donchian_high[i]:
                        position = 1
                        signals[i] = 0.25
                    elif close[i] < donchian_low[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals