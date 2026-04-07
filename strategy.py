#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and ADX trend filter
# Long when price breaks above 12h Donchian(20) high + 1-day volume > 1.5x 20-period average + ADX(14) > 20
# Short when price breaks below 12h Donchian(20) low + 1-day volume > 1.5x 20-period average + ADX(14) > 20
# Exit when price crosses Donchian midline or ADX < 18 (hysteresis)
# Stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses volume and ADX from daily timeframe for regime filtering
# Target: 80-160 total trades over 4 years (20-40/year)

name = "12h_donchian20_1d_vol_adx_v4"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / (volume_ma + 1e-10)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    # Calculate 1-day ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR and DM
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    
    atr_1d = tr_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * (plus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10))
    minus_di = 100 * (minus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 12-period Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 12-period ATR(14) for stoploss
    tr1_12 = high - low
    tr2_12 = np.abs(high - np.roll(close, 1))
    tr3_12 = np.abs(low - np.roll(close, 1))
    tr2_12[0] = tr1_12[0]
    tr3_12[0] = tr1_12[0]
    tr_12 = np.maximum(tr1_12, np.maximum(tr2_12, tr3_12))
    atr_12 = pd.Series(tr_12).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ratio_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(atr_12[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_12[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline or ADX < 18 (hysteresis)
            elif close[i] < donchian_mid[i] or adx_aligned[i] < 18:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_12[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midline or ADX < 18 (hysteresis)
            elif close[i] > donchian_mid[i] or adx_aligned[i] < 18:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and ADX confirmation
            # Volume filter: current volume > 1.5x 20-period average
            volume_confirmed = volume_ratio_aligned[i] > 1.5
            # Trend filter: ADX > 20
            trending = adx_aligned[i] > 20
            
            # Long: price breaks above Donchian high + volume + trend
            if close[i] > highest_high[i] and volume_confirmed and trending:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume + trend
            elif close[i] < lowest_low[i] and volume_confirmed and trending:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals