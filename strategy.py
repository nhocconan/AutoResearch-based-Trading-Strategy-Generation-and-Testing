#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour price action with 12-hour ADX trend filter
# Long when price breaks above 6h Donchian(20) high + 12h ADX > 25 (trending)
# Short when price breaks below 6h Donchian(20) low + 12h ADX > 25
# Exit when price crosses 6h Donchian midline OR ADX drops below 20 (trend weakening)
# Stoploss at 2 * ATR(14) to manage risk in volatile markets
# Position size: 0.25 (25% of capital)
# Uses 12-hour ADX for trend strength filtering to avoid whipsaws
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_donchian20_12h_adx_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12-hour data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12-hour ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_12h_s = pd.Series(tr_12h)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr_12h = tr_12h_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * (plus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h)
    minus_di = 100 * (minus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h)
    
    # Avoid division by zero
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    dx_s = pd.Series(dx)
    adx_12h = dx_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 6h
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6-hour Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 6-period ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses midline OR ADX drops below 20 (trend weakening)
            elif close[i] < donchian_mid[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses midline OR ADX drops below 20 (trend weakening)
            elif close[i] > donchian_mid[i] or adx_12h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with ADX trend filter
            # Trend filter: ADX > 25 indicates strong trend
            strong_trend = adx_12h_aligned[i] > 25
            
            # Long: price breaks above Donchian high + strong trend
            if close[i] > donchian_high[i] and strong_trend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + strong trend
            elif close[i] < donchian_low[i] and strong_trend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals