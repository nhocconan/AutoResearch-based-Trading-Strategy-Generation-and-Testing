#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, enter long when price breaks above 20-period Donchian high AND 1d trend is up (close > EMA50) AND volume > 2.0x 20-period average. Enter short when price breaks below 20-period Donchian low AND 1d trend is down (close < EMA50) AND volume spike. Uses Donchian channels for breakout structure, 1d EMA50 for responsive trend filter, and volume confirmation to reduce false breakouts. Designed for moderate trade frequency (12-37/year) with strong risk control via trend alignment and volatility filters. Targets BTC/ETH primarily.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Donchian Channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high = max(high, 20), Donchian low = min(low, 20)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (use previous 12h bar's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for volatility filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA warmup (50), Donchian warmup (20), volume MA warmup (20), ATR warmup (14)
    start_idx = max(50, 20, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # ATR filter: only trade when volatility is normal (not extreme)
        atr_mean = np.mean(atr[max(0, i-50):i]) if i > 0 else atr[i]
        atr_ratio = atr[i] / np.maximum(atr_mean, 1e-10)
        volatility_normal = (atr_ratio > 0.3) & (atr_ratio < 3.0)
        
        # Breakout conditions relative to Donchian levels
        breakout_above = close[i] > donchian_high_aligned[i]
        breakout_below = close[i] < donchian_low_aligned[i]
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price above Donchian high + 1d uptrend + volume spike + normal volatility
            long_signal = breakout_above and trend_uptrend and volume_spike[i] and volatility_normal
            
            # Short: price below Donchian low + 1d downtrend + volume spike + normal volatility
            short_signal = breakout_below and trend_downtrend and volume_spike[i] and volatility_normal
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR trend change to downtrend OR volatility extreme
            if close[i] < donchian_low_aligned[i] or not trend_uptrend or not volatility_normal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR trend change to uptrend OR volatility extreme
            if close[i] > donchian_high_aligned[i] or not trend_downtrend or not volatility_normal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0