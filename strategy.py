#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter.
# Long when price breaks above Donchian(20) high + volume > 1.5x average volume + price > 12h EMA50.
# Short when price breaks below Donchian(20) low + volume > 1.5x average volume + price < 12h EMA50.
# Exit when price crosses Donchian(20) midpoint or volume dries up.
# Uses discrete position size 0.25. Donchian provides clear breakout levels, volume confirms momentum,
# and 12h EMA50 ensures trading with higher timeframe trend to avoid whipsaws in ranging markets.
# 4h timeframe targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 4h Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian(20) high/low
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma_20 + low_ma_20) / 2.0
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h Indicators: EMA50 for trend filter ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, high_ma_20)  # dummy array for alignment - will be replaced
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, low_ma_20)   # dummy array for alignment - will be replaced
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid) # dummy array for alignment - will be replaced
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)         # dummy array for alignment - will be replaced
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Now compute the actual 4h indicators and align them properly
    df_4h = get_htf_data(prices, '4h')  # This is correct - we need 4h data for 4h indicators
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Recompute 4h indicators on actual 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Donchian(20) on 4h data
    high_ma_20_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_ma_20_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (high_ma_20_4h + low_ma_20_4h) / 2.0
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_ma_20_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_ma_20_4h)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_ma_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        donchian_high = donchian_high_aligned[i]
        donchian_low = donchian_low_aligned[i]
        donchian_mid = donchian_mid_aligned[i]
        vol_ma = vol_ma_aligned[i]
        vol_current = volume[i]
        ema50 = ema50_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian midpoint OR volume < 0.5x average volume (drying up)
            if (price < donchian_mid) or (vol_current < 0.5 * vol_ma):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian midpoint OR volume < 0.5x average volume (drying up)
            if (price > donchian_mid) or (vol_current < 0.5 * vol_ma):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian high + volume > 1.5x average volume + price > 12h EMA50
            if (price > donchian_high) and (vol_current > 1.5 * vol_ma) and (price > ema50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Donchian low + volume > 1.5x average volume + price < 12h EMA50
            elif (price < donchian_low) and (vol_current > 1.5 * vol_ma) and (price < ema50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_12hEMA50_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0