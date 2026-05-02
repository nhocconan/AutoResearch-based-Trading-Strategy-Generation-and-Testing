#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend + volume spike
# Uses Donchian channel breakouts on daily timeframe with weekly HMA trend filter
# Volume confirmation ensures breakouts have conviction
# Works in bull markets by buying breakouts above upper channel and in bear markets by selling breakdowns below lower channel
# Weekly HMA(21) provides smooth trend filter that adapts to both bull and bear regimes
# Targets 30-100 trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_1wHMA21_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high of last 20 days
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 days
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (wait for completed 1d bar)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    
    # Load 1w data ONCE before loop for HMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w HMA(21)
    close_1w = df_1w['close'].values
    hma_21 = calculate_hma(close_1w, 21)
    
    # Align HMA to 1d timeframe (wait for completed 1w bar)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper Donchian + price > weekly HMA + volume spike
            if close[i] > upper_20_aligned[i] and close[i] > hma_21_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + price < weekly HMA + volume spike
            elif close[i] < lower_20_aligned[i] and close[i] < hma_21_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian
            if close[i] < lower_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian
            if close[i] > upper_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    # WMA of full period
    wma_full = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma