#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and EMA(50) trend filter.
Uses 1w volume > 1.5x 50-period average and EMA(50) direction to filter breakouts.
Targets 15-25 trades/year to avoid fee drag. Works in bull (breakouts with volume) and bear (EMA filter avoids false breakouts).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === Daily Donchian Channel (20) ===
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly EMA(50) for trend direction ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Weekly volume confirmation ===
    volume_1w = df_1w['volume'].values
    vol_ma_50 = pd.Series(volume_1w).rolling(window=50, min_periods=50).mean().values
    vol_ma_50_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_50)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        
        # Volume spike: current 1w volume > 1.5x 50-period average
        df_1w_current = get_htf_data(prices, '1w')
        vol_1w_current = df_1w_current['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w_current, vol_1w_current)
        vol_spike = vol_1w_aligned[i] > vol_ma_50_aligned[i] * 1.5
        
        # Trend filter: price above/below EMA(50)
        price_above_ema = price > ema_50_1w_aligned[i]
        price_below_ema = price < ema_50_1w_aligned[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: Price breaks above Donchian upper + volume spike + price > EMA(50)
            if price > high_roll[i] and vol_spike and price_above_ema:
                signals[i] = 0.25
                position = 1
                continue
            # Short: Price breaks below Donchian lower + volume spike + price < EMA(50)
            elif price < low_roll[i] and vol_spike and price_below_ema:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal on opposite breakout
        elif position == 1:
            # Exit long if price breaks below Donchian lower
            if price < low_roll[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price breaks above Donchian upper
            if price > high_roll[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wVolume1.5x_EMA50_Trend"
timeframe = "1d"
leverage = 1.0