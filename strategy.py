#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20-period) with volume confirmation and 1d EMA200 trend filter.
# Uses 4h for signal direction (trend), 1h for entry timing precision.
# 1d EMA200 filters for long-term trend to avoid counter-trend entries.
# Volume > 1.5x 20-period EMA ensures institutional participation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
name = "1h_Donchian20_4hTrend_1dEMA200_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h Donchian channels: upper = max(high, 20), lower = min(low, 20)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate rolling max/min with period=20
    donchian_high = np.full_like(high_4h, np.nan)
    donchian_low = np.full_like(low_4h, np.nan)
    
    for i in range(20, len(high_4h)):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Shift by 1 to use only completed 4h bars (avoid look-ahead)
    donchian_high_shifted = np.roll(donchian_high, 1)
    donchian_low_shifted = np.roll(donchian_low, 1)
    donchian_high_shifted[0] = np.nan
    donchian_low_shifted[0] = np.nan
    
    # Align to 1h timeframe
    donchian_high_1h = align_htf_to_ltf(prices, df_4h, donchian_high_shifted)
    donchian_low_1h = align_htf_to_ltf(prices, df_4h, donchian_low_shifted)
    
    # 1d EMA200 trend filter
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume spike filter: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ema20)
    
    # Session filter: 08-20 UTC (only trade during active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # wait for EMA200 to be ready
    
    for i in range(start_idx, n):
        # Skip if required data unavailable or outside session
        if (np.isnan(donchian_high_1h[i]) or np.isnan(donchian_low_1h[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ema20[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above 4h Donchian high with volume spike and above 1d EMA200
            if (price > donchian_high_1h[i] and vol_spike[i] and price > ema_200_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian low with volume spike and below 1d EMA200
            elif (price < donchian_low_1h[i] and vol_spike[i] and price < ema_200_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 4h Donchian low (mean reversion)
            if price < donchian_low_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price rises back above 4h Donchian high (mean reversion)
            if price > donchian_high_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals