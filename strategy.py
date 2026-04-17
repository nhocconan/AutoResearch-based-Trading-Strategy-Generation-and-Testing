#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 1d EMA(34) trend filter + 1h volume spike (x2) + ATR(14) stop
Long: price breaks above Donchian high, price > EMA34, volume > 2x 1h MA
Short: price breaks below Donchian low, price < EMA34, volume > 2x 1h MA
Exit: opposite Donchian break or ATR-based stop
Target: 20-40 trades/year per symbol, avoids overtrading via strict volume and trend filters
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    donch_high = df_4h['high'].rolling(window=20, min_periods=20).max()
    donch_low = df_4h['low'].rolling(window=20, min_periods=20).min()
    donch_high_4h = align_htf_to_ltf(prices, df_4h, donch_high.values)
    donch_low_4h = align_htf_to_ltf(prices, df_4h, donch_low.values)
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1h volume spike filter (2x 24-period MA)
    df_1h = get_htf_data(prices, '1h')
    volume_ma_24 = pd.Series(df_1h['volume']).rolling(window=24, min_periods=24).mean()
    volume_ma_24_1h = align_htf_to_ltf(prices, df_1h, volume_ma_24.values)
    
    # ATR(14) for stop loss
    df_4h_atr = get_htf_data(prices, '4h')
    tr1 = df_4h_atr['high'] - df_4h_atr['low']
    tr2 = abs(df_4h_atr['high'] - df_4h_atr['close'].shift(1))
    tr3 = abs(df_4h_atr['low'] - df_4h_atr['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean()
    atr_4h = align_htf_to_ltf(prices, df_4h_atr, atr.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_4h[i]) or np.isnan(donch_low_4h[i]) or 
            np.isnan(ema_34_1d[i]) or np.isnan(volume_ma_24_1h[i]) or 
            np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24_1h[i]
        atr_val = atr_4h[i]
        
        if position == 0:
            # Long: break above Donchian high + trend filter + volume spike
            if price > donch_high_4h[i] and price > ema_34_1d[i] and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below Donchian low + trend filter + volume spike
            elif price < donch_low_4h[i] and price < ema_34_1d[i] and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: opposite break or ATR stop
            if price < donch_low_4h[i] or price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: opposite break or ATR stop
            if price > donch_high_4h[i] or price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_1dEMA34_1hVolSpike_ATRStop"
timeframe = "4h"
leverage = 1.0