#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h EMA crossover (8/21) with 4h Donchian breakout (20) and 1d volume spike filter.
# Long when 1h EMA8 crosses above EMA21, price > 4h Donchian upper (20), AND 1d volume > 1.5x 20-day average.
# Short when 1h EMA8 crosses below EMA21, price < 4h Donchian lower (20), AND 1d volume > 1.5x 20-day average.
# Exit on opposite EMA crossover or Donchian midpoint breach.
# Uses 4h/1d for signal direction/filters, 1h only for entry timing.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in both bull (breakouts with volume) and bear (short breakdowns with volume).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: EMA8 and EMA21 ===
    ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # EMA crossover signals (from previous bar to avoid look-ahead)
    ema8_prev = np.roll(ema8, 1)
    ema21_prev = np.roll(ema21, 1)
    ema8_prev[0] = np.nan
    ema21_prev[0] = np.nan
    
    ema8_cross_above = (ema8_prev <= ema21_prev) & (ema8 > ema21)
    ema8_cross_below = (ema8_prev >= ema21_prev) & (ema8 < ema21)
    
    # === 4h Indicators: Donchian Channel (20) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian upper/lower (20-period)
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donch_mid_4h = (donch_high_4h + donch_low_4h) / 2
    
    # Align to 1h timeframe (completed 4h bars only)
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    donch_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_mid_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-day average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(donch_high_4h_aligned[i]) or
            np.isnan(donch_low_4h_aligned[i]) or np.isnan(donch_mid_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        ema8_cross_up = ema8_cross_above[i]
        ema8_cross_down = ema8_cross_below[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if EMA8 crosses below EMA21 OR price breaks below Donchian midpoint
            if ema8_cross_down or price < donch_mid_4h_aligned[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if EMA8 crosses above EMA21 OR price breaks above Donchian midpoint
            if ema8_cross_up or price > donch_mid_4h_aligned[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: EMA8 crosses above EMA21, price > Donchian upper, AND volume spike
            if ema8_cross_up and price > donch_high_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: EMA8 crosses below EMA21, price < Donchian lower, AND volume spike
            elif ema8_cross_down and price < donch_low_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_EMA8_21_Cross_Donchian20_4h_VolumeSpike_1d_V1"
timeframe = "1h"
leverage = 1.0