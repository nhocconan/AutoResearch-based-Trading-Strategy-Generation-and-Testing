#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# Uses 12h Donchian channels for breakout detection, filtered by 1d EMA trend direction
# and confirmed by volume spikes. Works in both bull and bear markets by following
# the higher timeframe trend. Target: 15-30 trades/year to minimize fee decay.
# Volatility filter (ATR-based) avoids whipsaws in choppy markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Donchian channels (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_period = 20
    
    donchian_high = np.full(len(high_12h), np.nan)
    donchian_low = np.full(len(low_12h), np.nan)
    
    for i in range(donchian_period - 1, len(high_12h)):
        donchian_high[i] = np.max(high_12h[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low_12h[i-donchian_period+1:i+1])
    
    # Calculate 34-period EMA on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate ATR for volatility filter (14-period on 12h)
    atr_period = 14
    tr = np.zeros(len(high_12h))
    tr[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(high_12h)):
        tr[i] = max(
            high_12h[i] - low_12h[i],
            abs(high_12h[i] - close_12h[i-1]),
            abs(low_12h[i] - close_12h[i-1])
        )
    
    atr_12h = np.full(len(tr), np.nan)
    for i in range(atr_period - 1, len(tr)):
        if i == atr_period - 1:
            atr_12h[i] = np.mean(tr[:atr_period])
        else:
            atr_12h[i] = (atr_12h[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate average volume on 12h for spike detection
    vol_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(vol_12h), np.nan)
    vol_period = 10
    for i in range(vol_period, len(vol_12h)):
        vol_ma_12h[i] = np.mean(vol_12h[i-vol_period:i])
    
    # Align all indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(donchian_period, ema_len, atr_period, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_12h_aligned[i] if vol_ma_12h_aligned[i] > 0 else 0
        
        # Volatility filter: avoid extremes (both too low and too high)
        # Normal ATR range: 0.5 to 2.0 times the median
        atr_median = np.nanmedian(atr_12h_aligned[max(0, i-50):i+1])
        if np.isnan(atr_median) or atr_median == 0:
            volatility_filter = False
        else:
            atr_ratio = atr_12h_aligned[i] / atr_median
            volatility_filter = 0.5 <= atr_ratio <= 2.0
        
        # Volume confirmation: at least 1.3x average volume
        volume_confirmation = vol_ratio > 1.3
        
        if position == 0:
            # Long: Donchian breakout above upper band with uptrend and volume
            if price > donchian_high_aligned[i] and price > ema_1d_aligned[i] and volume_confirmation and volatility_filter:
                signals[i] = size
                position = 1
            # Short: Donchian breakout below lower band with downtrend and volume
            elif price < donchian_low_aligned[i] and price < ema_1d_aligned[i] and volume_confirmation and volatility_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below Donchian middle or trend reverses
            donchian_middle = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price < donchian_middle or price < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above Donchian middle or trend reverses
            donchian_middle = (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2
            if price > donchian_middle or price > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_Breakout_1dEMA_Trend_Volume"
timeframe = "12h"
leverage = 1.0