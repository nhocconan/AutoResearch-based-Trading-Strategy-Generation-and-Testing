#!/usr/bin/env python3
"""
12h Donchian Breakout with Volume Confirmation and ADX Filter
- Uses 12h timeframe for primary signals, 1d for ADX filter
- Long when price breaks above Donchian(20) high with volume > 1.5x 20-period volume MA and daily ADX > 25
- Short when price breaks below Donchian(20) low with volume > 1.5x 20-period volume MA and daily ADX > 25
- Exit when price crosses back through the opposite Donchian band or ADX drops below 20
- Position size: 0.25 to manage drawdown
- Target: 50-150 trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 12h volume confirmation (20-period MA)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d ADX filter (loaded once, aligned to 12h)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX on daily data
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / atr_1d)
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for Donchian (20) + volume MA (20) + ADX (14)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        adx = adx_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and ADX filter
            if price > donchian_high[i] and vol > 1.5 * vol_ma and adx > 25:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and ADX filter
            elif price < donchian_low[i] and vol > 1.5 * vol_ma and adx > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian low or ADX weak
            if price < donchian_low[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian high or ADX weak
            if price > donchian_high[i] or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0