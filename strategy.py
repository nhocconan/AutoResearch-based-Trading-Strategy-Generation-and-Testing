#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day ADX trend filter and volume confirmation
# Long when price breaks above Donchian upper(20) and daily ADX > 25 (strong trend)
# Short when price breaks below Donchian lower(20) and daily ADX > 25 (strong trend)
# Exit when price crosses opposite Donchian level or stoploss at 2.5 * ATR
# Volume confirmation: current volume > 1.8 * average volume of last 20 periods
# Position size: 0.28 (28% of capital)
# Target: 75-200 total trades over 4 years (19-50/year)
# Uses daily ADX to filter for strong trends that work in both bull and bear markets

name = "4h_donchian20_1d_adx_vol_v1"
timeframe = "4h"
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
    
    # 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(adx_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = position * 0.28
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower(20)
            elif close[i] < low[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.28
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper(20)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.28
        else:
            # Calculate Donchian channels (20-period)
            highest_high = high[i-20:i].max() if i >= 20 else high[:i].max()
            lowest_low = low[i-20:i].min() if i >= 20 else low[:i].min()
            
            # Trend filter: daily ADX > 25 for strong trend
            strong_trend = adx_aligned[i] > 25
            
            # Volume confirmation: current volume > 1.8 * average volume
            volume_confirm = volume[i] > 1.8 * vol_avg[i]
            
            # Long: price breaks above Donchian upper(20) in strong trend with volume
            if close[i] > highest_high and strong_trend and volume_confirm:
                signals[i] = 0.28
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower(20) in strong trend with volume
            elif close[i] < lowest_low and strong_trend and volume_confirm:
                signals[i] = -0.28
                position = -1
                entry_price = close[i]
    
    return signals