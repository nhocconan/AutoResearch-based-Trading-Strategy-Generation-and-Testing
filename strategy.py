#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian channel (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily indicators to daily timeframe (no alignment needed as we're on 1d)
    donch_high_aligned = donch_high
    donch_low_aligned = donch_low
    close_1d_aligned = close_1d
    volume_1d_aligned = volume_1d
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup for weekly EMA200
        # Skip if data not ready
        if (np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema200 = ema200_1w_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        price = close_1d_aligned[i]
        vol = volume_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter: volume > 1.5x average (avoid low volume false breakouts)
        vol_filter = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly EMA200 + volume
            if price > donch_high_val and price > ema200 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly EMA200 + volume
            elif price < donch_low_val and price < ema200 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Donchian opposite level
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low
                if price < donch_low_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high
                if price > donch_high_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA200_Trend_Volume"
timeframe = "1d"
leverage = 1.0