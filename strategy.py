#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRRegime
Hypothesis: On 4h timeframe, enter long when price breaks above Donchian(20) upper band AND 12h trend is up (close > EMA50) AND volume > 2.0x 20-period average AND ATR regime normal (0.5 < ATR/MA(ATR,50) < 2.0). Enter short when price breaks below Donchian(20) lower band AND 12h trend is down (close < EMA50) AND volume spike AND ATR regime normal. Uses Donchian breakouts with 12h EMA50 trend filter, volume confirmation, and ATR-based volatility regime filter. Designed for 20-50 trades/year with strong edge in both bull and bear markets via trend alignment and volatility filtering.
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian(20) channels on primary timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    # ATR for volatility regime filter (14-period ATR, 50-period MA)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.maximum(atr_ma, 1e-10)
    atr_regime_normal = (atr_ratio > 0.5) & (atr_ratio < 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian warmup (20), EMA50 warmup (50), volume MA warmup (20), ATR warmup (14+50)
    start_idx = max(20, 50, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions relative to Donchian channels
        breakout_above = close[i] > donchian_upper[i]
        breakout_below = close[i] < donchian_lower[i]
        
        # 12h trend filter
        trend_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price above Donchian upper + 12h uptrend + volume spike + normal ATR regime
            long_signal = breakout_above and trend_uptrend and volume_spike[i] and atr_regime_normal[i]
            
            # Short: price below Donchian lower + 12h downtrend + volume spike + normal ATR regime
            short_signal = breakout_below and trend_downtrend and volume_spike[i] and atr_regime_normal[i]
            
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
            # Exit: price breaks below Donchian lower OR trend change to downtrend OR ATR regime extreme
            if close[i] < donchian_lower[i] or not trend_uptrend or not atr_regime_normal[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR trend change to uptrend OR ATR regime extreme
            if close[i] > donchian_upper[i] or not trend_downtrend or not atr_regime_normal[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRRegime"
timeframe = "4h"
leverage = 1.0