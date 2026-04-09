#!/usr/bin/env python3
# 1d_1w_donchian_hma_regime_v1
# Hypothesis: Daily Donchian(20) breakout with 1-week HMA(21) trend filter and volume confirmation.
# Long: Price breaks above Donchian(20) high + weekly HMA(21) upward + volume > 1.5x 20-day average.
# Short: Price breaks below Donchian(20) low + weekly HMA(21) downward + volume > 1.5x 20-day average.
# Exit: Price crosses Donchian midpoint or opposite breakout with volume confirmation.
# Uses daily timeframe for signals, weekly for trend filter to avoid false breakouts in ranging markets.
# Target: 30-100 total trades over 4 years (7-25/year) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_hma_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Get 1w data for HMA trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly HMA(21)
    close_1w = df_1w['close'].values
    hma_21 = calculate_hma(close_1w, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Weekly HMA trend: slope over 3 periods
        if i >= 3:
            hma_slope = hma_21_aligned[i] - hma_21_aligned[i-3]
            hma_up = hma_slope > 0
            hma_down = hma_slope < 0
        else:
            hma_up = False
            hma_down = False
        
        if position == 1:  # Long position
            # Exit: Price crosses Donchian midpoint or breaks below Donchian low with volume
            if close[i] <= donchian_mid[i] or (close[i] < donchian_low[i] and volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price crosses Donchian midpoint or breaks above Donchian high with volume
            if close[i] >= donchian_mid[i] or (close[i] > donchian_high[i] and volume_confirmed):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout with volume confirmation and trend filter
            bullish_breakout = (close[i] > donchian_high[i]) and volume_confirmed and hma_up
            bearish_breakout = (close[i] < donchian_low[i]) and volume_confirmed and hma_down
            
            if bullish_breakout:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout:
                position = -1
                signals[i] = -0.25
    
    return signals

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    weights_half = np.arange(1, half_period + 1)
    wma_half = np.convolve(arr, weights_half, mode='valid') / weights_half.sum()
    wma_half = np.concatenate([np.full(half_period-1, np.nan), wma_half])
    
    # WMA of full period
    weights_full = np.arange(1, period + 1)
    wma_full = np.convolve(arr, weights_full, mode='valid') / weights_full.sum()
    wma_full = np.concatenate([np.full(period-1, np.nan), wma_full])
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt period
    weights_sqrt = np.arange(1, sqrt_period + 1)
    hma = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
    hma = np.concatenate([np.full(len(raw_hma)-len(hma), np.nan), hma])
    
    return hma