#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21) and volume confirmation.
Long when price breaks above Donchian upper band AND 1w HMA21 rising AND volume > 1.5x 20-period MA.
Short when price breaks below Donchian lower band AND 1w HMA21 falling AND volume > 1.5x 20-period MA.
Exit when price touches opposite Donchian band or 1w HMA21 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian provides clear structure, 1w HMA21 filters major trend, volume confirms breakout strength.
Works in bull (trend continuation) and bear (mean reversion via opposite breaks).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        # Use lookback of 20 periods (excluding current bar to avoid look-ahead)
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 1w HMA21 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full(len(values), np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Calculate WMA for half period
    wma_half = np.full(len(close_1w), np.nan)
    if len(close_1w) >= half_len:
        wma_half_vals = wma(close_1w, half_len)
        wma_half[half_len-1:half_len-1+len(wma_half_vals)] = wma_half_vals
    
    # Calculate WMA for full period
    wma_full = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        wma_full_vals = wma(close_1w, 21)
        wma_full[20:20+len(wma_full_vals)] = wma_full_vals
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = np.full(len(close_1w), np.nan)
    valid_idx = ~(np.isnan(wma_half) | np.isnan(wma_full))
    raw_hma[valid_idx] = 2 * wma_half[valid_idx] - wma_full[valid_idx]
    
    # Final HMA: WMA of raw_hma with sqrt(n) period
    hma_21_1w = np.full(len(close_1w), np.nan)
    if len(raw_hma) >= sqrt_len:
        # Find first valid index in raw_hma
        first_valid = np.argmax(~np.isnan(raw_hma)) if np.any(~np.isnan(raw_hma)) else len(raw_hma)
        if first_valid + sqrt_len <= len(raw_hma):
            valid_raw = raw_hma[first_valid:][~np.isnan(raw_hma[first_valid:])]
            if len(valid_raw) >= sqrt_len:
                hma_vals = wma(valid_raw, sqrt_len)
                hma_idx = first_valid + sqrt_len - 1
                hma_21_1w[hma_idx:hma_idx+len(hma_vals)] = hma_vals
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate 1d volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21, 20)  # Donchian (needs 20), HMA21, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        hma_val = hma_21_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate HMA slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            hma_prev = hma_21_aligned[i-1]
            hma_rising = hma_val > hma_prev
            hma_falling = hma_val < hma_prev
        else:
            hma_rising = False
            hma_falling = False
        
        # Volume filter: 1d volume > 1.5x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND HMA rising AND volume filter
            if price > upper and hma_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND HMA falling AND volume filter
            elif price < lower and hma_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower (opposite) OR HMA starts falling
                if price < lower or (i >= start_idx + 1 and hma_val < hma_21_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper (opposite) OR HMA starts rising
                if price > upper or (i >= start_idx + 1 and hma_val > hma_21_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wHMA21_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0