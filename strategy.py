#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
# Donchian breakout captures momentum in both bull and bear markets
# 1w HMA(21) ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation (1.5x 20-period average) ensures participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by using higher timeframe trend filter

name = "1d_Donchian20_1wHMA_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for HMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w HMA(21)
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    close_1w = df_1w['close'].values
    wma_half = wma(close_1w, half_len)
    wma_full = wma(close_1w, 21)
    # Pad to original length
    wma_half_padded = np.full_like(close_1w, np.nan)
    wma_full_padded = np.full_like(close_1w, np.nan)
    wma_half_padded[half_len-1:] = wma_half
    wma_full_padded[21-1:] = wma_full
    hma_raw = 2 * wma_half_padded - wma_full_padded
    hma_1w = wma(hma_raw, sqrt_len)
    # Pad HMA result
    hma_1w_padded = np.full_like(close_1w, np.nan)
    hma_1w_padded[sqrt_len-1:] = hma_1w[:len(close_1w)-sqrt_len+1]
    
    # Align 1w HMA to 1d
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    
    # Calculate 1d Donchian channels (20-period)
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate 1d volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = lookback  # 20 periods for Donchian + 1 for shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w HMA
        # Uptrend: close > HMA, Downtrend: close < HMA
        uptrend = close[i] > hma_1w_aligned[i]
        downtrend = close[i] < hma_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if uptrend:
                # In uptrend: look for long breakout above Donchian upper
                if high[i] > donchian_upper[i] and volume_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif downtrend:
                # In downtrend: look for short breakout below Donchian lower
                if low[i] < donchian_lower[i] and volume_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: price closes below Donchian lower or trend reverses
            if close[i] < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: price closes above Donchian upper or trend reverses
            if close[i] > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals