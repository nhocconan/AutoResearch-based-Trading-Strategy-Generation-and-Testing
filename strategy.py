#!/usr/bin/env python3
"""
4h 20-period Donchian Breakout with Volume Confirmation and 1D EMA Trend Filter
Long: Close breaks above Donchian high (20) + volume > 1.5x 4h volume MA(20) + close > 1D EMA50
Short: Close breaks below Donchian low (20) + volume > 1.5x 4h volume MA(20) + close < 1D EMA50
Exit: Opposite Donchian break (close crosses Donchian mid-line)
Volatility filter: Skip trades when 4h ATR(14) > 1.5x its 50-period MA (avoid extreme volatility)
Target: 20-30 trades/year per symbol
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 4h volume moving average (20-period)
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 4h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    # 1D EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 60  # warmup for longest indicator
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(volume_ma_20[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma_50[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20[i]
        
        # Volatility filter: avoid extreme volatility
        vol_filter = atr[i] <= 1.5 * atr_ma_50[i]
        
        if position == 0:
            # Long: break above Donchian high + volume + 1D trend + vol filter
            if (price > donchian_high[i] and vol > 1.5 * vol_ma and 
                price > ema_50_1d_aligned[i] and vol_filter):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low + volume + 1D trend + vol filter
            elif (price < donchian_low[i] and vol > 1.5 * vol_ma and 
                  price < ema_50_1d_aligned[i] and vol_filter):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: close below Donchian mid-line
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian mid-line
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_1DEMA50_VolFilter"
timeframe = "4h"
leverage = 1.0