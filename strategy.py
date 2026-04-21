#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for structure
    df_4h = get_htf_data(prices, '4h')
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike detection on 4h
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Price array
    close = prices['close'].values
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        ema50_daily = ema50_1d_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        price = close[i]
        
        # Trend filter: price above/below daily EMA50
        uptrend = price > ema50_daily
        downtrend = price < ema50_daily
        
        # Volume spike detection
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: price breaks above 4h Donchian high + daily uptrend + volume spike
            if price > donch_high_val and uptrend and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low + daily downtrend + volume spike
            elif price < donch_low_val and downtrend and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level
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
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Donchian20_DailyEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0