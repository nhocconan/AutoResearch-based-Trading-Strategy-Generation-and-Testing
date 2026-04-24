#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h HMA21 for trend direction (bullish if price > HMA21, bearish if price < HMA21).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to ensure strong participation.
- Entry: Long when price breaks above 20-bar Donchian high AND 12h HMA21 bullish AND volume spike.
         Short when price breaks below 20-bar Donchian low AND 12h HMA21 bearish AND volume spike.
- Exit: Opposite Donchian level (20-bar low for long, 20-bar high for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Donchian channels provide objective breakout levels. Combined with HMA trend filter (reduces lag vs EMA)
and strict volume confirmation, this avoids false breakouts and works in both bull and bear markets by
only taking trades in the direction of the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(arr).ewm(span=half_period, adjust=False, min_periods=half_period).mean().values
    # WMA of full period
    wma_full = pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values
    # Raw HMA
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean().values
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) using previous bar's data to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # 20-period rolling max/min of previous high/low
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Get 12h data for HMA21 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21
    df_12h_close = df_12h['close'].values
    hma_12h = calculate_hma(df_12h_close, 21)
    
    # Calculate 20-period volume MA on 12h
    df_12h_volume = df_12h['volume'].values
    vol_ma_12h = pd.Series(df_12h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_12h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 21, 20)  # Need enough bars for Donchian, HMA, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        hma_val = hma_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above Donchian high AND 12h HMA21 bullish (price > HMA)
                if curr_high > donchian_high[i] and curr_close > hma_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND 12h HMA21 bearish (price < HMA)
                elif curr_low < donchian_low[i] and curr_close < hma_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation
            if curr_low < donchian_low[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation
            if curr_high > donchian_high[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hHMA21_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0