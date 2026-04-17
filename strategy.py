#!/usr/bin/env python3
"""
4h Donchian(20) Breakout with Volume Spike and ADX Trend Filter
Long: Price breaks above Donchian(20) high + volume > 2x 4h volume MA + ADX > 25
Short: Price breaks below Donchian(20) low + volume > 2x 4h volume MA + ADX > 25
Exit: Opposite Donchian break
ATR-based stop loss: exit if price moves against position by 2.5x ATR(14)
Target: 20-40 trades/year per symbol (80-160 total over 4 years)
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
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average (20-period for confirmation)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # ATR for stop loss (14-period)
    atr_sl = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(adx[i]) or np.isnan(atr_sl[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = volume_ma[i]
        
        if position == 0:
            # Long: break above Donchian high + volume spike + trend
            if price > donch_high[i] and vol > 2.0 * vol_ma_val and adx[i] > 25:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low + volume spike + trend
            elif price < donch_low[i] and vol > 2.0 * vol_ma_val and adx[i] > 25:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: opposite Donchian break or stop loss
            if price < donch_low[i] or price < entry_price - 2.5 * atr_sl[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite Donchian break or stop loss
            if price > donch_high[i] or price > entry_price + 2.5 * atr_sl[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ADX25_ATRStop"
timeframe = "4h"
leverage = 1.0