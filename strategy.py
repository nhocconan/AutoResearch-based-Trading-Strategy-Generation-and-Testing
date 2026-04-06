#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w ADX filter and volume confirmation
# Long when price breaks above 1d Donchian upper band, 1w ADX > 25 (trending), and volume > 1.5x monthly average
# Short when price breaks below 1d Donchian lower band, 1w ADX > 25 (trending), and volume > 1.5x monthly average
# Exit when ADX < 20 (range) or opposite breakout occurs
# Stoploss at 2.0 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses 1w ADX for trend strength filter and monthly volume to confirm breakout validity
# Target: 30-100 total trades over 4 years (7-25/year)

name = "1d_donchian20_1w_adx_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d Donchian(20) channels
    high_series = pd.Series(high_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    low_series = pd.Series(low_1d)
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # 1w ADX for trend strength filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX components
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # 1m volume for confirmation (using monthly approximation)
    df_1m = get_htf_data(prices, '1m')
    if len(df_1m) < 20:
        return np.zeros(n)
    
    volume_1m = df_1m['volume'].values
    volume_ma_1m = pd.Series(volume_1m).rolling(window=20, min_periods=20).mean().values
    volume_ma_1m_aligned = align_htf_to_ltf(prices, df_1m, volume_ma_1m)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_1m_aligned[i]) or 
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
            # Exit: ADX weakens (<20) or breaks below lower band
            elif adx_aligned[i] < 20 or close[i] < lower_aligned[i]:
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
            # Exit: ADX weakens (<20) or breaks above upper band
            elif adx_aligned[i] < 20 or close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and strong trend (ADX > 25)
            # Long: price breaks above upper band, ADX > 25 (strong trend), volume spike
            if (close[i] > upper_aligned[i] and
                adx_aligned[i] > 25 and
                volume[i] > 1.5 * volume_ma_1m_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower band, ADX > 25 (strong trend), volume spike
            elif (close[i] < lower_aligned[i] and
                  adx_aligned[i] > 25 and
                  volume[i] > 1.5 * volume_ma_1m_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals