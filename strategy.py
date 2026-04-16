#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 6h HMA(21) trend filter and volume confirmation.
# Long when price breaks above 1d Donchian upper AND close > 6h HMA21 AND volume > 1.5x 20-period MA.
# Short when price breaks below 1d Donchian lower AND close < 6h HMA21 AND volume > 1.5x 20-period MA.
# Exit when price returns to 1d Donchian midpoint OR HMA21 cross reverses.
# Uses discrete position size 0.25. Donchian provides structure, HMA filters noise, volume confirms strength.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Get 6h data once before loop for HMA and volume MA
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d Indicators: Donchian(20) channels ===
    # Upper = max(high, 20), Lower = min(low, 20), Middle = (upper + lower)/2
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window, min_periods=window).min().values
    
    donch_20_upper = rolling_max(high_1d, 20)
    donch_20_lower = rolling_min(low_1d, 20)
    donch_20_middle = (donch_20_upper + donch_20_lower) / 2
    
    # === 6h Indicators: HMA(21) and Volume MA(20) ===
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(arr, weights/weights.sum(), mode='same')
    
    def hma(arr, window):
        half = window // 2
        sqrt_n = int(np.sqrt(window))
        wma_half = wma(arr, half)
        wma_full = wma(arr, window)
        raw_hma = 2 * wma_half - wma_full
        return wma(raw_hma, sqrt_n)
    
    hma21_6h = hma(close_6h, 21)
    vol_ma20_6h = pd.Series(volume_6h).rolling(20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donch_20_upper_aligned = align_htf_to_ltf(prices, df_1d, donch_20_upper)
    donch_20_lower_aligned = align_htf_to_ltf(prices, df_1d, donch_20_lower)
    donch_20_middle_aligned = align_htf_to_ltf(prices, df_1d, donch_20_middle)
    hma21_6h_aligned = align_htf_to_ltf(prices, df_6h, hma21_6h)
    vol_ma20_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma20_6h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # Donchian20 needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_20_upper_aligned[i]) or np.isnan(donch_20_lower_aligned[i]) or 
            np.isnan(donch_20_middle_aligned[i]) or np.isnan(hma21_6h_aligned[i]) or 
            np.isnan(vol_ma20_6h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        donch_upper = donch_20_upper_aligned[i]
        donch_lower = donch_20_lower_aligned[i]
        donch_middle = donch_20_middle_aligned[i]
        hma21 = hma21_6h_aligned[i]
        vol_ma = vol_ma20_6h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_confirm = vol > 1.5 * vol_ma
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price <= Donchian middle OR HMA cross turns bearish (price < HMA)
            if (price <= donch_middle) or (price < hma21):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price >= Donchian middle OR HMA cross turns bullish (price > HMA)
            if (price >= donch_middle) or (price > hma21):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND price > HMA21 AND volume confirmation
            if (price > donch_upper) and (price > hma21) and volume_confirm:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below Donchian lower AND price < HMA21 AND volume confirmation
            elif (price < donch_lower) and (price < hma21) and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_6hHMA21_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0