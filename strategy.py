#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper channel + price > 12h EMA50 + volume spike
# Short when price breaks below 4h Donchian lower channel + price < 12h EMA50 + volume spike
# Exit when price returns to 4h Donchian middle or trend reverses
# Designed for low trade frequency (~25-50/year) with edge in trending markets
# Uses proven structure from top performers: price channel + trend + volume

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 20-period Donchian channels on 4h
    high_max_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_max_20 + low_min_20) / 2.0
    
    # Align Donchian levels to 4h timeframe (already aligned but ensuring proper timing)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, high_max_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, low_min_20)
    donch_mid_aligned = align_htf_to_ltf(prices, df_4h, mid_20)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        donch_high = donch_high_aligned[i]
        donch_low = donch_low_aligned[i]
        donch_mid = donch_mid_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume spike
            if price > donch_high and price > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume spike
            elif price < donch_low and price < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian middle or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to Donchian middle or trend turns down
                if price <= donch_mid or price < ema_trend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to Donchian middle or trend turns up
                if price >= donch_mid or price > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Volume_Spike"
timeframe = "4h"
leverage = 1.0