#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with volume confirmation and ATR stop
# - Long when price breaks above 20-period Donchian high with volume > 1.3x 20-period average
# - Short when price breaks below 20-period Donchian low with volume > 1.3x 20-period average
# - Exit when price crosses back through midline or ATR-based stop hit (2*ATR)
# - Uses 1d ADX for trend filter: only trade when ADX > 25 (trending market)
# - Target: 20-35 trades per year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX
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
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # 4h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (20-period)
    atr_tr1 = high - low
    atr_tr2 = np.abs(high - np.roll(close, 1))
    atr_tr3 = np.abs(low - np.roll(close, 1))
    atr_tr2[0] = atr_tr1[0]
    atr_tr3[0] = atr_tr1[0]
    atr_tr = np.maximum(atr_tr1, np.maximum(atr_tr2, atr_tr3))
    atr_20 = pd.Series(atr_tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(atr_20[i]) or np.isnan(adx_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume surge + trending
            if price > donch_high[i] and vol > 1.3 * vol_ma[i] and trending:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + volume surge + trending
            elif price < donch_low[i] and vol > 1.3 * vol_ma[i] and trending:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below midline OR ATR stop hit (2*ATR)
            if price < donch_mid[i] or price < entry_price - 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midline OR ATR stop hit (2*ATR)
            if price > donch_mid[i] or price > entry_price + 2.0 * atr_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_ADXFilter_ATRStop"
timeframe = "4h"
leverage = 1.0