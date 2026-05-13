#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high, 12h HMA21 is rising (uptrend), and volume > 1.8x 20-period average.
# Short when price breaks below 20-period Donchian low, 12h HMA21 is falling (downtrend), and volume > 1.8x 20-period average.
# Uses ATR(14) trailing stop (2.5x) for risk control.
# Donchian channels provide robust price channels that adapt to volatility, working in both trending and ranging markets.
# 12h HMA21 trend filter reduces whipsaws by ensuring trades align with intermediate-term momentum.
# Volume confirmation validates breakout strength. Target: 75-200 total trades over 4 years (19-50/year) on 4h.

name = "4h_Donchian20_HMA21_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate HMA(21) on 12h data: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def calculate_hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        if half_window < 1 or sqrt_window < 1:
            return np.full_like(values, np.nan)
        wma_half = calculate_wma(values, half_window)
        wma_full = calculate_wma(values, window)
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(values, np.nan)
        raw_hma = 2 * wma_half - wma_full
        hma = calculate_wma(raw_hma, sqrt_window)
        # Pad to original length
        result = np.full_like(values, np.nan)
        start_idx = window - half_window  # Account for WMA padding
        end_idx = start_idx + len(hma)
        if end_idx <= len(values) and start_idx >= 0:
            result[start_idx:end_idx] = hma
        return result
    
    hma_21_12h = calculate_hma(close_12h, 21)
    
    # Align 12h HMA21 to 4h timeframe (wait for 12h bar to close)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Donchian high AND 12h HMA21 rising (trending up) AND volume confirmation
            if close[i] > donchian_high[i] and hma_21_12h_aligned[i] > hma_21_12h_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Donchian low AND 12h HMA21 falling (trending down) AND volume confirmation
            elif close[i] < donchian_low[i] and hma_21_12h_aligned[i] < hma_21_12h_aligned[i-1] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.25
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals