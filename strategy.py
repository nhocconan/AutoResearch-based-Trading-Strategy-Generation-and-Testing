#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with 1d ADX > 25 and volume > 1.5x average.
# Short when price breaks below 20-period Donchian low with 1d ADX > 25 and volume > 1.5x average.
# Exit when price crosses the 10-period Donchian midpoint (mean reversion).
# Uses Donchian for clear breakout signals, ADX for trend strength, volume for confirmation.
# Designed to work in both bull (breakouts) and bear (breakdowns) markets with controlled trade frequency.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    adx_period = 14
    tr = np.maximum(np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1])), np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr])  # Align with index
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    atr = np.full(len(tr), np.nan)
    plus_di = np.full(len(tr), np.nan)
    minus_di = np.full(len(tr), np.nan)
    dx = np.full(len(tr), np.nan)
    adx = np.full(len(tr), np.nan)
    
    if len(tr) >= adx_period:
        # Initial ATR
        atr[adx_period - 1] = np.nanmean(tr[1:adx_period])
        for i in range(adx_period, len(tr)):
            atr[i] = (atr[i-1] * (adx_period - 1) + tr[i]) / adx_period
        
        # Directional indicators
        plus_di = 100 * plus_dm / atr
        minus_di = 100 * minus_dm / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX smoothing
        adx[adx_period * 2 - 2] = np.nanmean(dx[adx_period-1:adx_period*2-1])
        for i in range(adx_period * 2 - 1, len(dx)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    donch_period_entry = 20
    donch_period_exit = 10
    
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    highest_high_exit = np.full(n, np.nan)
    lowest_low_exit = np.full(n, np.nan)
    
    for i in range(donch_period_entry - 1, n):
        highest_high[i] = np.max(high[i - donch_period_entry + 1:i + 1])
        lowest_low[i] = np.min(low[i - donch_period_entry + 1:i + 1])
    
    for i in range(donch_period_exit - 1, n):
        highest_high_exit[i] = np.max(high[i - donch_period_exit + 1:i + 1])
        lowest_low_exit[i] = np.min(low[i - donch_period_exit + 1:i + 1])
    
    # Donchian midpoint for exit
    donch_mid = (highest_high_exit + lowest_low_exit) / 2
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, ADX, and volume MA20
    start_idx = max(donch_period_entry, adx_period * 2 - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(donch_mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        # Trend filter: require ADX > 25 (strong trend)
        trend_filter = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with trend and volume
            if price > highest_high[i] and trend_filter and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below Donchian low with trend and volume
            elif price < lowest_low[i] and trend_filter and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint
            if price < donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint
            if price > donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dADX14_Volume"
timeframe = "12h"
leverage = 1.0