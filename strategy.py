#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 12h HMA21 for trend filter (price > HMA21 = uptrend, price < HMA21 = downtrend).
- Entry: Long when price breaks above Donchian(20) high AND price > 12h HMA21 AND volume > 1.5 * 4h volume MA(20);
         Short when price breaks below Donchian(20) low AND price < 12h HMA21 AND volume > 1.5 * 4h volume MA(20).
- Exit: Long exits when price breaks below Donchian(20) low; Short exits when price breaks above Donchian(20) high.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying breakouts in uptrend) and bear (selling breakdowns in downtrend) with volume confirmation to avoid false breakouts.
- Uses Donchian channel for structure and HMA for smooth trend following.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average."""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half = arr.rolling(window=period//2, min_periods=period//2).mean()
    sqrt = arr.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    raw = 2 * half - arr
    hma = pd.Series(raw).rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

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
    if len(df_12h) < 21:  # Need sufficient data for HMA21
        return np.zeros(n)
    
    # Calculate HMA21 for 12h
    close_12h = df_12h['close'].values
    hma_21 = calculate_hma(close_12h, 21)
    
    # Align HMA21 to 4h timeframe
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # Calculate Donchian(20) channels on 4h
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume MA(20) for 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_21_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold
        vol_confirm = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: price breaks above Donchian high AND price > 12h HMA21 (uptrend)
                if curr_high > donch_high[i] and curr_close > hma_21_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low AND price < 12h HMA21 (downtrend)
                elif curr_low < donch_low[i] and curr_close < hma_21_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price breaks below Donchian low
            if curr_low < donch_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price breaks above Donchian high
            if curr_high > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hHMA21_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0