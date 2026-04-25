#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop
Hypothesis: Donchian(20) breakouts on 4h capture strong momentum. Trend filter from 12h EMA50 ensures alignment with higher timeframe direction. Volume spike confirms breakout strength. ATR-based stoploss limits drawdown. Works in bull markets (breakouts continuation) and bear markets (breakdown continuation). Uses discrete position sizing (0.25) to limit fee drag and drawdown. Target: 20-50 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_donchian(high, low, window):
    """Calculate Donchian channel upper and lower bands."""
    upper = pd.Series(high).rolling(window=window, min_periods=window).max().values
    lower = pd.Series(low).rolling(window=window, min_periods=window).min().values
    return upper, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for EMA50 trend filter (loaded ONCE)
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicator to LTF (4h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian(20) for breakout signals
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # 4h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # 4h ATR for dynamic stoploss
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start index: need Donchian (20) + volume MA (20) + ATR (14) + aligned HTF arrays
    start_idx = max(20, 20, 14, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume spike and 12h uptrend
            long_breakout = (curr_close > donchian_upper[i]) and vol_spike[i] and (curr_close > ema_50_12h_aligned[i])
            # Short: price breaks below Donchian lower with volume spike and 12h downtrend
            short_breakout = (curr_close < donchian_lower[i]) and vol_spike[i] and (curr_close < ema_50_12h_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price breaks below Donchian lower OR trend turns down OR ATR stoploss hit
            if (curr_close < donchian_lower[i]) or (curr_close < ema_50_12h_aligned[i]) or (curr_close < entry_price - 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price breaks above Donchian upper OR trend turns up OR ATR stoploss hit
            if (curr_close > donchian_upper[i]) or (curr_close > ema_50_12h_aligned[i]) or (curr_close > entry_price + 2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0