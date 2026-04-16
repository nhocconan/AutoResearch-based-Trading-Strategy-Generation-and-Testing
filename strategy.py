#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Supertrend trend direction with 1d Donchian breakout and volume confirmation.
# Long when 1w Supertrend is bullish, price breaks above Donchian(20) upper band, and volume > 1.5x 20-period average.
# Short when 1w Supertrend is bearish, price breaks below Donchian(20) lower band, and volume > 1.5x 20-period average.
# Exit when 1w Supertrend flips direction or price crosses Donchian midline.
# Uses discrete position size 0.25. Supertrend filters trend, Donchian provides breakout signals, volume confirms strength.
# Target: 30-100 total trades over 4 years (7-25/year) for BTC/ETH/SOL.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # === 1w Indicators: Supertrend (ATR=10, mult=3.0) ===
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2.0
    upper_band = hl2 + (3.0 * atr_1w)
    lower_band = hl2 - (3.0 * atr_1w)
    
    # Supertrend calculation
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i-1] > upper_band[i-1]:
            direction[i] = 1
        elif close_1w[i-1] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(upper_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(lower_band[i], supertrend[i-1])
    
    # Supertrend direction (1=uptrend, -1=downtrend)
    supertrend_dir = direction
    
    # Align 1w Supertrend and direction to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_dir)
    
    # Get 1d data once before loop for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Donchian Channel (20-period)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume moving average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        st_dir = supertrend_dir_aligned[i]
        price = close[i]
        vol = volume[i]
        dh = donchian_high[i]
        dl = donchian_low[i]
        dm = donchian_mid[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Supertrend turns bearish or price crosses below Donchian midline
            if st_dir <= 0 or price < dm:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Supertrend turns bullish or price crosses above Donchian midline
            if st_dir >= 0 or price > dm:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Supertrend bullish, price breaks above Donchian high, volume confirmation
            if (st_dir > 0) and (price > dh) and vol_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Supertrend bearish, price breaks below Donchian low, volume confirmation
            elif (st_dir < 0) and (price < dl) and vol_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_1wSupertrend_Donchian20_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0