#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(20) breakout with 1-day volume confirmation and ADX trend filter
# Long when price breaks above 12h Donchian(20) high + 1-day volume > 1.5x 20-period average + 1-day ADX > 25
# Short when price breaks below 12h Donchian(20) low + 1-day volume > 1.5x 20-period average + 1-day ADX > 25
# Exit when price crosses Donchian midpoint or ADX < 20
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Designed for trending markets with volume confirmation to avoid false breakouts
# Target: 75-150 total trades over 4 years (19-38/year)

name = "12h_donchian20_1d_vol_adx_v2"
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
    
    # 1-day data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day volume average (20-period)
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume_1d / (volume_ma + 1e-10)
    
    # Calculate 1-day ADX (14-period)
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
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM
    tr_s = pd.Series(tr)
    plus_dm_s = pd.Series(plus_dm)
    minus_dm_s = pd.Series(minus_dm)
    atr_1d = tr_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * (plus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10))
    minus_di_1d = 100 * (minus_dm_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / (atr_1d + 1e-10))
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d + 1e-10)
    adx_1d = pd.Series(dx_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 12-period Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1-day indicators to 12h
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 12-period ATR(14) for stoploss
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.roll(close, 1))
    tr3_12h = np.abs(low - np.roll(close, 1))
    tr2_12h[0] = tr1_12h[0]
    tr3_12h[0] = tr1_12h[0]
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(atr_12h[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or weak trend
            elif close[i] < donchian_mid[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_12h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses Donchian midpoint or weak trend
            elif close[i] > donchian_mid[i] or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for entries: Donchian breakout with volume and trend confirmation
            volume_confirm = volume_ratio_aligned[i] > 1.5
            trend_confirm = adx_1d_aligned[i] > 25
            
            # Long: price breaks above Donchian high + volume + trend
            if close[i] > donchian_high[i] and volume_confirm and trend_confirm:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low + volume + trend
            elif close[i] < donchian_low[i] and volume_confirm and trend_confirm:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals