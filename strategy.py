#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h EMA trend direction with 1h Donchian breakout and volume confirmation.
# Long when 4h EMA(34) is rising, price breaks above 1h Donchian(20) high, and volume > 1.5x 20-period average.
# Short when 4h EMA(34) is falling, price breaks below 1h Donchian(20) low, and volume > 1.5x 20-period average.
# Exit when 4h EMA direction reverses or price returns to the Donchian midpoint.
# Uses discrete position size 0.20. EMA provides smooth trend, Donchian captures breakouts,
# volume filter ensures conviction. Target: 60-150 total trades over 4 years (15-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA(34) ===
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_dir_4h = np.zeros_like(ema_34_4h)
    ema_34_dir_4h[1:] = np.where(ema_34_4h[1:] > ema_34_4h[:-1], 1, np.where(ema_34_4h[1:] < ema_34_4h[:-1], -1, 0))
    
    # Align 4h EMA and direction to 1h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    ema_34_dir_aligned = align_htf_to_ltf(prices, df_4h, ema_34_dir_4h)
    
    # Get 1h data once before loop for Donchian and volume
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    volume_1h = df_1h['volume'].values
    
    # Donchian Channel (20-period) on 1h
    donchian_high_20 = pd.Series(high_1h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align 1h Donchian to 1h timeframe (no alignment needed as we're using 1h data directly)
    donchian_high_aligned = donchian_high_20
    donchian_low_aligned = donchian_low_20
    donchian_mid_aligned = donchian_mid
    
    # Volume moving average (20-period) on 1h
    vol_ma_20_1h = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = vol_ma_20_1h  # no alignment needed
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(ema_34_dir_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        ema_val = ema_34_aligned[i]
        ema_dir_val = ema_34_dir_aligned[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        donchian_mid = donchian_mid_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if EMA direction turns down or price returns to Donchian midpoint
            if ema_dir_val <= 0 or price <= donchian_mid:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if EMA direction turns up or price returns to Donchian midpoint
            if ema_dir_val >= 0 or price >= donchian_mid:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: EMA direction must be non-zero
            trend_filter = ema_dir_val != 0
            
            # Volume filter: volume > 1.5x 20-period average (1h)
            vol_filter = vol > 1.5 * vol_ma_val
            
            # Breakout filters
            breakout_long = price > donchian_high
            breakout_short = price < donchian_low
            
            # LONG: EMA up, price breaks above Donchian high, volume spike
            if (ema_dir_val > 0) and breakout_long and vol_filter:
                signals[i] = 0.20
                position = 1
            
            # SHORT: EMA down, price breaks below Donchian low, volume spike
            elif (ema_dir_val < 0) and breakout_short and vol_filter:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_4hEMA34_1hDonchian20_VolumeSpike_V1"
timeframe = "1h"
leverage = 1.0