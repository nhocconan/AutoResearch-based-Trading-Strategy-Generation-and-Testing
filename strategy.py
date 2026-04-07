#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day volume confirmation and 1-week ADX trend filter
# Long when price breaks above Donchian(20) high + 1-day volume > 1.5x 20-day average + 1-week ADX > 25 (trending market)
# Short when price breaks below Donchian(20) low + 1-day volume > 1.5x 20-day average + 1-week ADX > 25
# Exit when price crosses Donchian midline or ADX < 20 (trend weakening)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_vol_1w_adx_v1"
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
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-week data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1-day volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_1d_s = pd.Series(vol_1d)
    vol_ma_20 = vol_1d_s.rolling(window=20, min_periods=20).mean().values
    vol_ratio = vol_1d / (vol_ma_20 + 1e-10)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # 1-week ADX (14-period)
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
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_s = pd.Series(tr_1w)
    atr_1w = tr_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    plus_di_1w = 100 * (plus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10))
    minus_di_1w = 100 * (minus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1w + 1e-10))
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w + 1e-10)
    dx_s = pd.Series(dx)
    adx_1w = dx_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 12-hour Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 12-hour ATR(14) for stoploss
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
            np.isnan(vol_ratio_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
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
            # Exit: price crosses Donchian midline or ADX < 20 (trend weakening)
            elif close[i] < donchian_mid[i] or adx_1w_aligned[i] < 20:
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
            # Exit: price crosses Donchian midline or ADX < 20 (trend weakening)
            elif close[i] > donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with volume and ADX trend filter
            vol_filter = vol_ratio_aligned[i] > 1.5
            adx_filter = adx_1w_aligned[i] > 25
            
            # Long: price breaks above Donchian high + volume + ADX trend
            if close[i] > donchian_high[i] and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume + ADX trend
            elif close[i] < donchian_low[i] and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals