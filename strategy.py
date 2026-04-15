#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume confirmation + ADX trend filter
# Donchian breakout captures trend continuation with clear entry/exit levels
# Volume confirmation ensures breakouts have institutional participation
# ADX > 25 filters for trending markets only, avoiding false breakouts in ranges
# Designed for low trade frequency (target 15-25/year) with high win rate
# Works in bull markets (breakouts continue) and bear markets (breakdowns continue)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 12h Donchian(20) channels
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        donchian_high[i] = np.max(high[i-lookback:i])
        donchian_low[i] = np.min(low[i-lookback:i])
    
    # 1d ADX(14) for trend filter
    # True Range
    tr1 = np.maximum(high_1d[1:], low_1d[:-1]) - np.minimum(high_1d[1:], low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Directional Indicators
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 1d volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(volume[i])):
            continue
        
        # Only trade in trending markets (ADX > 25)
        if adx_aligned[i] > 25:
            # Volume confirmation: current volume > 1.5x 20-day average
            volume_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
            
            # Long breakout: price closes above Donchian high
            if close[i] > donchian_high_aligned[i] and volume_confirm and position <= 0:
                position = 1
                signals[i] = position_size
            # Short breakdown: price closes below Donchian low
            elif close[i] < donchian_low_aligned[i] and volume_confirm and position >= 0:
                position = -1
                signals[i] = -position_size
            # Exit when price returns to midpoint (reversion signal)
            elif position == 1 and close[i] < (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
            elif position == -1 and close[i] > (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0