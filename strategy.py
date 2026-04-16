#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian Breakout with 1w Trend Filter and Volume Confirmation
# Uses Donchian(20) breakout on daily chart as entry signal, confirmed by 1w EMA200 trend and 1.5x volume spike.
# Exits on opposite Donchian(10) touch or volatility contraction.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data (primary timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w data (trend filter) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # === Donchian Channels (20 for entry, 10 for exit) ===
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    donchian_low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # === 1w EMA200 trend filter ===
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    uptrend_1w = close_1w > ema200_1w  # Calculate on 1w
    uptrend_1w_aligned = align_htf_to_ltf(prices, df_1w, uptrend_1w.astype(float))
    
    # === Volume confirmation (1.5x 20-day average) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_spike = volume_1d > (1.5 * vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: max of Donchian(20), EMA200(200)
    warmup = 200
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_20[i]) or 
            np.isnan(donchian_low_20[i]) or
            np.isnan(ema200_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        dh20 = donchian_high_20[i]
        dl20 = donchian_low_20[i]
        dh10 = donchian_high_10[i]
        dl10 = donchian_low_10[i]
        trend_up = uptrend_1w_aligned[i] > 0.5
        vol_spike_val = vol_spike[i]
        
        # === STOPLOSS via opposing Donchian(10) touch ===
        if position == 1:  # Long position
            if price <= dl10:  # Touch or breach lower 10-period Donchian
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            if price >= dh10:  # Touch or breach upper 10-period Donchian
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long: price breaks above 20-period Donchian + uptrend + volume spike
            if price > dh20 and trend_up and vol_spike_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Short: price breaks below 20-period Donchian + downtrend + volume spike
            elif price < dl20 and not trend_up and vol_spike_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_DonchianBreakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0