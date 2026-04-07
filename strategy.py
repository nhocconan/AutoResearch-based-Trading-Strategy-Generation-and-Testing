#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(40) breakout with weekly volume confirmation and monthly ADX trend filter
# Long when price breaks above 40-day Donchian high + weekly volume > 1.3x 4-week average + monthly ADX > 20
# Short when price breaks below 40-day Donchian low + weekly volume > 1.3x 4-week average + monthly ADX > 20
# Exit when price crosses opposite Donchian level
# Stoploss at 2.0 * ATR(20)
# Position size: 0.25 (25% of capital)
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian40_1w_vol_1m_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-week data for volume confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 4:
        return np.zeros(n)
    
    # 1-month data for ADX trend filter
    df_1m = get_htf_data(prices, '1m')
    if len(df_1m) < 20:
        return np.zeros(n)
    
    # Calculate 1-week volume average (4-period)
    volume_1w = df_1w['volume'].values
    volume_1w_s = pd.Series(volume_1w)
    volume_ma_4w = volume_1w_s.rolling(window=4, min_periods=4).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_4w)
    
    # Calculate 1-month ADX (14-period)
    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    
    # True Range
    tr1 = high_1m - low_1m
    tr2 = np.abs(high_1m - np.roll(close_1m, 1))
    tr3 = np.abs(low_1m - np.roll(close_1m, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1m = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1m, prepend=high_1m[0])
    down_move = np.diff(low_1m, prepend=low_1m[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1m).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / (tr_14 + 1e-10)
    minus_di = 100 * minus_dm_14 / (tr_14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1m, adx)
    
    # 40-period Donchian channels
    highest_high = pd.Series(high).rolling(window=40, min_periods=40).max().values
    lowest_low = pd.Series(low).rolling(window=40, min_periods=40).min().values
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian low
            elif close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian high
            elif close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and ADX filter
            # Volume filter: volume > 1.3x 4-week average
            volume_filter = volume[i] > 1.3 * volume_ma_aligned[i]
            # Trend filter: monthly ADX > 20
            trend_filter = adx_aligned[i] > 20
            
            # Long: price breaks above Donchian high + volume filter + trend filter
            if close[i] > highest_high[i] and volume_filter and trend_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume filter + trend filter
            elif close[i] < lowest_low[i] and volume_filter and trend_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals