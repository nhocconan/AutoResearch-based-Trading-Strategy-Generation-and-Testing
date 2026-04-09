#!/usr/bin/env python3
# 1d_donchian_breakout_volume_chop_v1
# Hypothesis: 1d strategy using Donchian(20) breakout on primary timeframe for entry,
# with 1w HTF trend filter (HMA21) and volume confirmation (>1.5x 20-period average).
# Choppiness regime filter: only trade when CHOP(14) < 61.8 (trending market).
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 10-25 trades/year.
# Uses 1w HTF data for trend, called ONCE before loop.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for HMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate HMA21 on weekly close
    close_1w = df_1w['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(arr, window):
        if len(arr) < window:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, window + 1, dtype=float)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_21 = wma(raw_hma, sqrt_len)
    
    # Pad to original length
    hma_21_padded = np.full_like(close_1w, np.nan)
    hma_21_padded[half_len + sqrt_len - 1:] = hma_21
    
    # Align to daily timeframe
    hma_21w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
    
    # Daily indicators
    # Donchian(20)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14)
    def choppiness_index(high, low, close, window=14):
        atr = pd.Series(np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))),
                                   np.abs(low - np.roll(close, 1)))).rolling(window=window, min_periods=1).sum()
        max_high = pd.Series(high).rolling(window=window, min_periods=1).max()
        min_low = pd.Series(low).rolling(window=window, min_periods=1).min()
        chop = 100 * np.log10(atr / (max_high - min_low)) / np.log10(window)
        return chop.values
    
    chop = choppiness_index(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(hma_21w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Choppiness regime filter: only trade when CHOP < 61.8 (trending market)
        chop_filter = chop[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian low OR weekly trend turns bearish
            if close[i] < donch_low[i] or close[i] < hma_21w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR weekly trend turns bullish
            if close[i] > donch_high[i] or close[i] > hma_21w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_confirmed and chop_filter:
                # Long entry: price breaks above Donchian high AND above weekly HMA (uptrend)
                if close[i] > donch_high[i] and close[i] > hma_21w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price breaks below Donchian low AND below weekly HMA (downtrend)
                elif close[i] < donch_low[i] and close[i] < hma_21w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals