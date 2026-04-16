#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian(20) breakout from 1d + volume confirmation + ATR filter.
# Long when price breaks above 1d Donchian upper channel AND volume > 1.5x 20-period average AND ATR(14) < ATR(50) (low volatility breakout).
# Short when price breaks below 1d Donchian lower channel AND volume > 1.5x 20-period average AND ATR(14) < ATR(50).
# Exit when price crosses 1d Donchian middle (20-period SMA of high/low) or ATR(14) > 2.0 * ATR(50) (high volatility stop).
# Uses discrete position size 0.25. Donchian channels provide structure, volume confirms conviction, ATR filter avoids choppy markets.
# 12h timeframe targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns) with volatility filter to avoid false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # === 1d Indicators: Donchian(20) channels ===
    # Upper channel = max(high, 20)
    # Lower channel = min(low, 20)
    # Middle channel = SMA of (upper + lower)/2, 20
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # === 1d Indicators: ATR(14) and ATR(50) for volatility filter ===
    # ATR = average of true ranges
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    atr50_aligned = align_htf_to_ltf(prices, df_1d, atr50)
    
    # Volume confirmation: 20-period average volume on 12h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # Donchian20 and ATR50 need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(atr14_aligned[i]) or 
            np.isnan(atr50_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_middle = donchian_middle_aligned[i]
        atr14 = atr14_aligned[i]
        atr50 = atr50_aligned[i]
        vol_ma = vol_ma20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian middle (trend break) OR high volatility (ATR14 > 2.0 * ATR50)
            if (price < donchian_middle) or (atr14 > 2.0 * atr50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian middle (trend break) OR high volatility (ATR14 > 2.0 * ATR50)
            if (price > donchian_middle) or (atr14 > 2.0 * atr50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Donchian upper AND volume > 1.5x volume MA AND low volatility (ATR14 < ATR50)
            if (price > donchian_upper) and (vol > 1.5 * vol_ma) and (atr14 < atr50):
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Donchian lower AND volume > 1.5x volume MA AND low volatility (ATR14 < ATR50)
            elif (price < donchian_lower) and (vol > 1.5 * vol_ma) and (atr14 < atr50):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_1dDonchian20_VolumeConfirmation_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0