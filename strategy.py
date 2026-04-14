#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h ADX trend filter + 12h Donchian breakout with volume confirmation
# Long when ADX > 25 (trending) AND price breaks above 12h Donchian upper channel AND volume > 1.5x average
# Short when ADX > 25 AND price breaks below 12h Donchian lower channel AND volume > 1.5x average
# Exit when price crosses 12h Donchian midline
# ADX filters for trending markets, Donchian provides clear breakout levels, volume confirms institutional interest
# Designed for 4h timeframe with 12h HTF for breakout context - targets 20-40 trades/year

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate ADX on 4h (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr = np.insert(tr, 0, 0)
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    # Calculate Donchian channels on 12h (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max()
    donch_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min()
    donch_mid = (donch_high + donch_low) / 2
    
    # Align 12h Donchian levels to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high.values)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low.values)
    donch_mid_aligned = align_htf_to_ltf(prices, df_12h, donch_mid.values)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        donch_mid_val = donch_mid_aligned[i]
        
        if position == 0:
            # Long setup: ADX > 25 (trending) AND price breaks above Donchian upper channel AND volume confirmation
            if (adx_val > 25 and close_val > donch_high_val and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: ADX > 25 AND price breaks below Donchian lower channel AND volume confirmation
            elif (adx_val > 25 and close_val < donch_low_val and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if close_val < donch_mid_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if close_val > donch_mid_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_ADX_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0