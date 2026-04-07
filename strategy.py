#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day volume confirmation and 1-week ADX trend filter
# Long when price breaks above 4h Donchian upper (20) + 1-day volume > 1.5x 20-day average + 1-week ADX > 25
# Short when price breaks below 4h Donchian lower (20) + 1-day volume > 1.5x 20-day average + 1-week ADX > 25
# Exit when price returns to Donchian midpoint or ADX < 20
# Stoploss at 2.5 * ATR(14)
# Position size: 0.30 (30% of capital)
# Uses daily volume and weekly ADX to filter breakouts in trending markets
# Target: 75-200 total trades over 4 years (19-50/year)

name = "4h_donchian20_1d_vol_1w_adx_v1"
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
    
    # 1-day data for volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1-week data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1-day volume 20-day average
    volume_1d = df_1d['volume'].values
    volume_1d_s = pd.Series(volume_1d)
    volume_ma_20 = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    
    # Calculate 1-week ADX (14)
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
    
    # 4h Donchian channels (20)
    lookback = 20
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_upper[i] = np.max(high[i-lookback+1:i+1])
        donchian_lower[i] = np.min(low[i-lookback+1:i+1])
        donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # 4h ATR(14) for stoploss
    tr1_4h = high - low
    tr2_4h = np.abs(high - np.roll(close, 1))
    tr3_4h = np.abs(low - np.roll(close, 1))
    tr2_4h[0] = tr1_4h[0]
    tr3_4h[0] = tr1_4h[0]
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_ma_20_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(atr_4h[i])):
            if position != 0:
                signals[i] = position * 0.30
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to midpoint or weak trend
            elif close[i] <= donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price returns to midpoint or weak trend
            elif close[i] >= donchian_mid[i] or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
        else:
            # Look for breakouts with volume confirmation and strong trend
            volume_surge = volume[i] > 1.5 * volume_ma_20_aligned[i]
            strong_trend = adx_1w_aligned[i] > 25
            
            # Long breakout
            if close[i] > donchian_upper[i] and volume_surge and strong_trend:
                signals[i] = 0.30
                position = 1
                entry_price = close[i]
            # Short breakdown
            elif close[i] < donchian_lower[i] and volume_surge and strong_trend:
                signals[i] = -0.30
                position = -1
                entry_price = close[i]
    
    return signals