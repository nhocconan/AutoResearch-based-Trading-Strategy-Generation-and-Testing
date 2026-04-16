#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + volume confirmation + choppiness regime filter.
# Long when price breaks above Donchian upper AND volume > 1.5x volume MA(20) AND chop > 61.8 (range).
# Short when price breaks below Donchian lower AND volume > 1.5x volume MA(20) AND chop > 61.8 (range).
# Exit when price crosses Donchian midline (median of upper/lower) OR chop < 38.2 (trend).
# Uses discrete position size 0.25. Choppiness filter avoids false breakouts in strong trends.
# Volume confirmation ensures breakouts have conviction. Targets 75-200 total trades over 4 years.
# Works in bull markets (catch breakouts) and bear markets (catch breakdowns) via range reversion.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for choppiness filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 4h Indicators: Donchian(20) and volume MA(20) ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    # Donchian midline = (upper + lower) / 2
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Indicators: Choppiness Index(14) ===
    # True Range = max(high-low, abs(high-close_prev), abs(low-close_prev))
    # Sum of TR over 14 periods
    # Choppiness = 100 * log10(sumTR / (ATR14 * 14)) / log10(14)
    # Simplified: CHOP = 100 * log10( sumTR / (true_range * 14) ) / log10(14)
    # Where true_range = max(high_1d) - min(low_1d) over 14 periods
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(tr1, np.abs(low_1d - np.roll(close_1d, 1)))
    tr = tr2
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr / (hh - ll)) / np.log10(14)
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), donchian_mid)
    volume_ma_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), volume_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50  # Donchian(20) + chop needs sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_mid = donchian_mid_aligned[i]
        vol_ma = volume_ma_aligned[i]
        chop = chop_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian midline OR chop < 38.2 (trending)
            if (price < donchian_mid) or (chop < 38.2):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian midline OR chop < 38.2 (trending)
            if (price > donchian_mid) or (chop < 38.2):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian upper AND volume > 1.5x volume MA AND chop > 61.8 (range)
            if (price > donchian_upper) and (vol > 1.5 * vol_ma) and (chop > 61.8):
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian lower AND volume > 1.5x volume MA AND chop > 61.8 (range)
            elif (price < donchian_lower) and (vol > 1.5 * vol_ma) and (chop > 61.8):
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_ChopFilter_V1"
timeframe = "4h"
leverage = 1.0