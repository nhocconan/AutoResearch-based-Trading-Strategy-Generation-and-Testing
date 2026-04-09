#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d HTF for signal direction and 1h for entry timing
# Uses 4h Donchian channel breakout with volume confirmation and 1d chop regime filter
# In choppy markets (CHOP > 61.8): mean reversion at Donchian bands
# In trending markets (CHOP < 38.2): breakout continuation
# Position size 0.20 to limit drawdown
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
# Session filter: 08-20 UTC to avoid low-volume periods
# Works in both bull/bear: adapts to regime via chop filter

name = "1h_4h_1d_donchian_chop_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    donchian_high = np.full(len(df_4h), np.nan)
    donchian_low = np.full(len(df_4h), np.nan)
    
    for i in range(20, len(df_4h)):
        donchian_high[i] = np.max(high_4h[i-19:i+1])
        donchian_low[i] = np.min(low_4h[i-19:i+1])
    
    # Calculate 1d Chop Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr0 = high_1d[i] - low_1d[i]
        tr1 = abs(high_1d[i] - close_1d[i-1])
        tr2 = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    # Sum of True Range over 14 periods
    tr_sum_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        tr_sum_14[i] = np.sum(tr[i-13:i+1])
    
    # Highest high and lowest low over 14 periods
    hh_14 = np.full(len(df_1d), np.nan)
    ll_14 = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        hh_14[i] = np.max(high_1d[i-13:i+1])
        ll_14[i] = np.min(low_1d[i-13:i+1])
    
    # Chop Index formula: 100 * log10(sum(TR14) / (HH14 - LL14)) / log10(14)
    chop = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        if hh_14[i] > ll_14[i]:
            chop[i] = 100 * np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    
    # Align HTF data to 1h timeframe
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low)
    chop_1h = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    # Calculate volume moving average (20-period) on 1h
    volume_ma = np.full(n, np.nan)
    for i in range(20, n):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # Start after warmup
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_1h[i]) or 
            np.isnan(donchian_low_1h[i]) or 
            np.isnan(chop_1h[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period MA
        vol_confirm = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions based on regime
            if chop_1h[i] > 61.8:  # Choppy regime - mean reversion
                # Exit when price reaches middle of Donchian channel
                mid = (donchian_high_1h[i] + donchian_low_1h[i]) / 2
                if close[i] >= mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
            else:  # Trending regime - breakout continuation
                # Exit when price closes below Donchian low
                if close[i] < donchian_low_1h[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.20
                    
        elif position == -1:  # Short position
            # Exit conditions based on regime
            if chop_1h[i] > 61.8:  # Choppy regime - mean reversion
                # Exit when price reaches middle of Donchian channel
                mid = (donchian_high_1h[i] + donchian_low_1h[i]) / 2
                if close[i] <= mid:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
            else:  # Trending regime - breakout continuation
                # Exit when price closes above Donchian high
                if close[i] > donchian_high_1h[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.20
        else:  # Flat
            # Entry logic based on regime and volume confirmation
            if vol_confirm:
                if chop_1h[i] > 61.8:  # Choppy regime - mean reversion
                    # Go long at Donchian low with rejection
                    # Go short at Donchian high with rejection
                    if low[i] <= donchian_low_1h[i] and close[i] > donchian_low_1h[i]:
                        position = 1
                        signals[i] = 0.20
                    elif high[i] >= donchian_high_1h[i] and close[i] < donchian_high_1h[i]:
                        position = -1
                        signals[i] = -0.20
                else:  # Trending regime - breakout continuation
                    # Go long on breakout above Donchian high
                    # Go short on breakdown below Donchian low
                    if high[i] > donchian_high_1h[i]:
                        position = 1
                        signals[i] = 0.20
                    elif low[i] < donchian_low_1h[i]:
                        position = -1
                        signals[i] = -0.20
    
    return signals