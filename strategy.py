#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w ADX trend filter and volume confirmation
# Breakouts above Donchian(20) high or below Donchian(20) low capture momentum
# 1w ADX(14) > 25 filters for trending markets where breakouts are more reliable
# Volume > 1.5x 20-period EMA confirms institutional participation
# Target: 7-25 trades/year with trend-following logic suited for 2025 bear/range conditions
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
    
    # Calculate 1w ADX (14-period) for trend filter (high ADX = trending)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range for ADX
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr2_1w[0] = 0
    tr3_1w[0] = 0
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    
    # Directional Movement
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1w = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    minus_di_1w = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned 1w ADX
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)[i]
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(adx_1w_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        # Trend filter: ADX > 25 indicates trending market (better for breakouts)
        trending = adx_1w_aligned > 25
        
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

name = "1d_1w_ADX_TrendBreak"
timeframe = "1d"
leverage = 1.0