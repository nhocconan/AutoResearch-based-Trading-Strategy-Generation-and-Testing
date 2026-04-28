#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility regime filter + 1w Supertrend direction + price breakout from 4h Donchian(20) channels.
# Uses 1d ATR ratio (short/long) to detect low volatility regimes where breakouts are more reliable.
# 1w Supertrend provides strong trend filter that works in both bull/bear markets.
# Donchian(20) breakouts on 4h capture medium-term momentum with fewer false signals than smaller periods.
# Volume confirmation (>1.5x 20-bar average) ensures breakout strength.
# Position size 0.25 balances return and drawdown. Target: 60-120 total trades over 4 years (15-30/year).

name = "4h_Donchian20_1wSupertrend_1dATRRegime_Volume_v1"
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
    
    # Get 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w Supertrend for trend filter (ATR=10, multiplier=3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR
    atr_1w = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2.0
    upper_band = hl2 + (3.0 * atr_1w)
    lower_band = hl2 - (3.0 * atr_1w)
    
    supertrend = np.zeros_like(close_1w)
    direction = np.ones_like(close_1w)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate 1d ATR ratio for volatility regime (short ATR / long ATR)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First value
    
    # Short-term ATR (7-period) and long-term ATR (30-period)
    atr_7 = pd.Series(tr_1d).ewm(span=7, adjust=False, min_periods=7).mean().values
    atr_30 = pd.Series(tr_1d).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # ATR ratio: low values indicate low volatility regime (good for breakouts)
    atr_ratio = atr_7 / atr_30
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Calculate 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or 
            np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1w Supertrend direction
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Volatility regime filter: low volatility regime (ATR ratio < 0.4)
        low_volatility = atr_ratio_aligned[i] < 0.4
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_high[i] and volume_spike[i]
        short_breakout = close[i] < donchian_low[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < donchian_low[i] or direction_aligned[i] == -1
        short_exit = close[i] > donchian_high[i] or direction_aligned[i] == 1
        
        # Handle entries and exits
        if long_breakout and uptrend and low_volatility and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and downtrend and low_volatility and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals