#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(55) breakout with weekly volume confirmation and weekly ADX trend filter
# Long when price breaks above 55-period Donchian high + weekly volume > 1.5x 55-period average + weekly ADX > 25
# Short when price breaks below 55-period Donchian low + weekly volume > 1.5x 55-period average + weekly ADX > 25
# Exit when price crosses 10-day EMA in opposite direction
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Uses weekly data for volume and trend confirmation to reduce noise and improve reliability
# Target: 50-100 total trades over 4 years (12-25/year)

name = "1d_donchian55_1w_vol_adx_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for volume confirmation and ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate weekly volume average (55-period)
    volume_1w = df_1w['volume'].values
    volume_1w_s = pd.Series(volume_1w)
    volume_ma = volume_1w_s.rolling(window=55, min_periods=55).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_1w, volume_ma)
    
    # Calculate weekly ADX (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1  # invert to positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_20 = pd.Series(tr_1w).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    plus_dm_20 = pd.Series(plus_dm).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    minus_dm_20 = pd.Series(minus_dm).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_20 / (tr_20 + 1e-10)
    minus_di = 100 * minus_dm_20 / (tr_20 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 55-period Donchian channels
    highest_high = pd.Series(high).rolling(window=55, min_periods=55).max().values
    lowest_low = pd.Series(low).rolling(window=55, min_periods=55).min().values
    
    # 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
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
    
    for i in range(55, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(ema_10[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below 10-day EMA
            elif close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above 10-day EMA
            elif close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume confirmation and ADX filter
            # Volume filter: volume > 1.5x 55-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: weekly ADX > 25
            trend_filter = adx_aligned[i] > 25
            
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