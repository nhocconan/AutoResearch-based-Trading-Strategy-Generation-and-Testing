#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and ATR(14) volatility filter
# Donchian channels provide clear breakout levels that work in both trending and ranging markets
# 12h HMA21 ensures we only trade breakouts in the direction of the higher timeframe trend
# ATR filter avoids low volatility environments where false breakouts are common
# Discrete sizing 0.25 minimizes fee churn. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_12hHMA21_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime errors
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop (MTF Rule #1)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 (Hull Moving Average)
    close_12h = df_12h['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    # HMA calculation
    wma_half = np.array([wma(close_12h[i:i+half_len], half_len) if i+half_len <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    wma_full = np.array([wma(close_12h[i:i+21], 21) if i+21 <= len(close_12h) else np.nan 
                         for i in range(len(close_12h))])
    hma_21_raw = 2 * wma_half - wma_full
    hma_21 = np.array([wma(hma_21_raw[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(hma_21_raw) else np.nan 
                       for i in range(len(hma_21_raw))])
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First period
    atr_14 = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 14)  # warmup for Donchian, ATR, and session
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(hma_21_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
            
        # Session filter: only trade 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_hma_21_12h = hma_21_aligned[i]
        curr_atr_14 = atr_14[i]
        
        # Volatility filter: avoid low volatility environments
        atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
        if np.isnan(atr_ma_50[i]) or curr_atr_14 < 0.5 * atr_ma_50[i]:
            # Low volatility - reduce position or stay flat
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish entry: break above Donchian high AND above 12h HMA21 (uptrend)
            if curr_high > curr_donchian_high and curr_close > curr_hma_21_12h:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Bearish entry: break below Donchian low AND below 12h HMA21 (downtrend)
            elif curr_low < curr_donchian_low and curr_close < curr_hma_21_12h:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price drops below Donchian low (breakout fails)
            if curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price rises above Donchian high (breakdown fails)
            if curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals