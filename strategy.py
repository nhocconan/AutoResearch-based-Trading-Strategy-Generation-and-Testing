#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w HMA21 trend filter and volume confirmation
# Uses 1d timeframe for signal generation with HTF 1w for trend alignment
# Volume confirmation (1.5x 20-period average) ensures institutional participation
# Regime filter: Chop < 61.8 (trending market) avoids false signals in ranging markets
# ATR-based stoploss (2.5x ATR) manages risk via position reduction to 0
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Works in bull markets via trend-aligned breakouts, in bear via chop regime filter
# Designed for low trade frequency to minimize fee drag

name = "1d_Donchian20_1wHMA21_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1w data ONCE before loop for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # Calculate WMA for half length
    if len(close_1w) >= half_len:
        wma_half = np.array([np.nan] * (half_len - 1) + list(wma(close_1w, half_len)))
    else:
        wma_half = np.full(len(close_1w), np.nan)
    
    # Calculate WMA for full length
    if len(close_1w) >= 21:
        wma_full = np.array([np.nan] * 20 + list(wma(close_1w, 21)))
    else:
        wma_full = np.full(len(close_1w), np.nan)
    
    # Calculate 2*WMA(half) - WMA(full)
    wma_diff = 2 * wma_half - wma_full
    
    # Calculate WMA of the difference with sqrt length
    if len(wma_diff) >= sqrt_len:
        hma_21 = np.array([np.nan] * (sqrt_len - 1) + list(wma(wma_diff, sqrt_len)))
    else:
        hma_21 = np.full(len(wma_diff), np.nan)
    
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # Load 1d data ONCE before loop for Chop regime filter and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (14) - trending when < 38.2, ranging when > 61.8
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr1 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log15(atr1 * 14 / (max_high - min_low))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    atr = atr1  # Use 1d ATR for stoploss
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(hma_21_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Chop < 61.8 (not strongly ranging)
        if chop_aligned[i] > 61.8:
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian High + price > 1w HMA21 + volume confirm
            if close[i] > donchian_high_aligned[i] and close[i] > hma_21_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian Low + price < 1w HMA21 + volume confirm
            elif close[i] < donchian_low_aligned[i] and close[i] < hma_21_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: Price drops below Donchian Low - 2.5*ATR
            if close[i] < donchian_low_aligned[i] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: Price breaks below Donchian High (trailing exit)
            elif close[i] < donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: Price rises above Donchian High + 2.5*ATR
            if close[i] > donchian_high_aligned[i] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: Price breaks above Donchian Low (trailing exit)
            elif close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals