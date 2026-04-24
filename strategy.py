#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 12h HMA21 for trend direction (bullish if close > HMA21, bearish if close < HMA21).
- Donchian channels: Upper = 20-period high, Lower = 20-period low on 4h.
- Entry: Long when price breaks above prior Donchian upper AND 12h HMA21 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below prior Donchian lower AND 12h HMA21 bearish AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 12h HMA21,
        exit short when price crosses above 12h HMA21.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures breakouts aligned with the medium-term trend, using volume to filter false breakouts.
Designed to work in both bull and bear markets by following the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h HMA21 for trend filter
    df_12h_close = df_12h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    wma_half = wma(df_12h_close, half_len)
    wma_full = wma(df_12h_close, 21)
    # Pad to same length
    wma_half_padded = np.full_like(df_12h_close, np.nan)
    wma_half_padded[half_len-1:] = wma_half
    wma_full_padded = np.full_like(df_12h_close, np.nan)
    wma_full_padded[20:] = wma_full
    hma_12h = 2 * wma_half_padded - wma_full_padded
    hma_12h = wma(hma_12h, sqrt_len)
    hma_12h_padded = np.full_like(df_12h_close, np.nan)
    hma_12h_padded[sqrt_len-1:sqrt_len-1+len(hma_12h)] = hma_12h
    
    # Calculate 4h Donchian channels (20-period)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # Need enough bars for Donchian and Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(donch_upper[i]) or 
            np.isnan(donch_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior Donchian upper AND 12h HMA21 bullish AND volume confirmed
            if curr_close > donch_upper[i] and curr_close > hma_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior Donchian lower AND 12h HMA21 bearish AND volume confirmed
            elif curr_close < donch_lower[i] and curr_close < hma_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 12h HMA21 (trend change)
            if curr_close < hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 12h HMA21 (trend change)
            if curr_close > hma_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0