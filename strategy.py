#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d ADX Filter and Volume Spike
# - Long when price breaks above 4h Donchian upper (20-period) + 1d ADX > 25 + volume > 2x 20-period average
# - Short when price breaks below 4h Donchian lower (20-period) + 1d ADX > 25 + volume > 2x 20-period average
# - Exit when price crosses back through Donchian midpoint or ATR-based stop hit (2x ATR)
# - Uses 1d for ADX trend filter (stable trend strength) and 4h for breakout execution
# - Target: 20-40 trades per year per symbol (80-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR for ADX and stop loss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DI and -DI for ADX
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_sum = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_di_sum = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * plus_di_sum / tr_sum
    minus_di = 100 * minus_di_sum / tr_sum
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate ATR for stop loss (using 1d data)
    atr_1d_for_stop = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_1d_for_stop_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_for_stop)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(vol_ma[i]) or np.isnan(adx_1d_aligned[i]) or np.isnan(atr_1d_for_stop_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + ADX > 25 + volume surge
            if price > donch_high[i] and adx_1d_aligned[i] > 25 and vol > 2.0 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below Donchian low + ADX > 25 + volume surge
            elif price < donch_low[i] and adx_1d_aligned[i] > 25 and vol > 2.0 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR ATR stop hit (2*ATR)
            if price < donch_mid[i] or price < entry_price - 2.0 * atr_1d_for_stop_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR ATR stop hit (2*ATR)
            if price > donch_mid[i] or price > entry_price + 2.0 * atr_1d_for_stop_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ADX25_VolumeSpike"
timeframe = "4h"
leverage = 1.0