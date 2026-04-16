#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d volume confirmation and session filter.
# Long when price breaks above 4h Donchian(20) high, 1d volume > 1.5x 20-day average, and UTC hour 8-20.
# Short when price breaks below 4h Donchian(20) low, 1d volume > 1.5x 20-day average, and UTC hour 8-20.
# Exit when price returns to 4h Donchian midpoint or session ends.
# Uses discrete position size 0.20. Donchian provides structure, volume confirms breakout validity,
# session filter avoids low-liquidity hours. Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian channels (20-period)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align 4h Donchian to 1h timeframe (waits for completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Get 1d data once before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # 1d volume moving average (20-period)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Session filter: UTC 8-20 only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        dh = donchian_high_aligned[i]
        dl = donchian_low_aligned[i]
        dm = donchian_mid_aligned[i]
        vol_ma = vol_ma_aligned[i]
        price = close[i]
        vol_1d = volume_1d[i // 24] if i // 24 < len(volume_1d) else volume_1d[-1]  # approximate 1d volume for current bar
        
        # Volume filter: current 1d volume > 1.5x 20-day average
        vol_filter = vol_1d > 1.5 * vol_ma
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to Donchian midpoint or breaks below lower band
            if price <= dm or price < dl:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to Donchian midpoint or breaks above upper band
            if price >= dm or price > dh:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above Donchian high with volume confirmation
            if price > dh and vol_filter:
                signals[i] = 0.20
                position = 1
            
            # SHORT: price breaks below Donchian low with volume confirmation
            elif price < dl and vol_filter:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_4hDonchian_1dVolBreakout_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0