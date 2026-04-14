#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ADX trend filter and volume confirmation
# Breakouts above Donchian(20) high or below Donchian(20) low capture momentum
# 1d ADX(14) > 25 filters for trending markets where breakouts are more reliable
# Volume > 1.5x 20-period EMA confirms institutional participation
# Target: 20-30 trades/year with trend-following logic suited for 2025 bear/range conditions
# Stops via opposite Donchian band touch to avoid whipsaws

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ADX (14-period) for trend filter (high ADX = trending)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for ADX
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    
    # Directional Movement
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1d ADX
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)[i]
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_1d_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        # Trend filter: ADX > 25 indicates trending market (better for breakouts)
        trending = adx_1d_aligned > 25
        
        if position == 0:  # No position - look for breakout entries
            # Long breakout: price breaks above Donchian high with volume in trending market
            if close[i] > donchian_high[i] and volume_confirm and trending:
                position = 1
                signals[i] = position_size
            # Short breakout: price breaks below Donchian low with volume in trending market
            elif close[i] < donchian_low[i] and volume_confirm and trending:
                position = -1
                signals[i] = -position_size
        elif position == 1:  # Long position - exit at opposite Donchian band
            # Exit if price breaks below Donchian low (failed breakout)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:  # Short position - exit at opposite Donchian band
            # Exit if price breaks above Donchian high (failed breakout)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_1dADX_TrendBreak"
timeframe = "4h"
leverage = 1.0