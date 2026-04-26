#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_4hTrend_VolumeChopFilter
Hypothesis: On 4h timeframe, use Donchian(20) breakouts filtered by 4h trend (close > EMA34) and choppiness regime (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend following). Enter long on upper band breakout with uptrend or range, short on lower band breakout with downtrend or range. Uses discrete position size 0.25 to limit fee drag. Designed for 20-50 trades/year on 4h by requiring trend alignment and volume/chop filters, reducing overtrading while capturing structured moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF filters (trend and chop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate Donchian channels on 4h (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper band: highest high over past 20 periods
    high_series = pd.Series(high_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    
    # Lower band: lowest low over past 20 periods
    low_series = pd.Series(low_4h)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 4h timeframe (no additional delay needed as they're based on completed 4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # Calculate 4h EMA34 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_34_4h = close_4h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate choppiness index on 4h (14-period)
    # CHOP = 100 * log10(sum(ATR over period) / (max(high) - min(low))) / log10(period)
    tr1 = np.abs(high_4h - low_4h)
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period ATR
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.maximum(max_high - min_low, 1e-10)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 4h EMA warmup, Donchian warmup, chop warmup
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 4h trend alignment
        trend_uptrend = close[i] > ema_34_4h_aligned[i]
        trend_downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Choppiness regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
        chop_value = chop_aligned[i]
        is_range = chop_value > 61.8
        is_trending = chop_value < 38.2
        
        if position == 0:
            # Long: price breaks above upper band + (uptrend OR range) + volume spike
            long_signal = (close[i] > donchian_upper_aligned[i]) and (trend_uptrend or is_range) and volume_spike[i]
            
            # Short: price breaks below lower band + (downtrend OR range) + volume spike
            short_signal = (close[i] < donchian_lower_aligned[i]) and (trend_downtrend or is_range) and volume_spike[i]
            
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
            # Exit: price breaks below lower band OR trend turns down AND chop < 38.2 (strong trend)
            if (close[i] < donchian_lower_aligned[i] or 
                (not trend_uptrend and chop_value < 38.2)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above upper band OR trend turns up AND chop < 38.2 (strong trend)
            if (close[i] > donchian_upper_aligned[i] or 
                (not trend_downtrend and chop_value < 38.2)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_4hTrend_VolumeChopFilter"
timeframe = "4h"
leverage = 1.0