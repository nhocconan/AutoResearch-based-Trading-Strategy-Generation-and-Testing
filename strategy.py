#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d EMA50 trend filter, volume confirmation (2.0x average), and ATR(14) stoploss (2.5x).
Long when price breaks above Donchian(20) upper band AND price > 1d EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian(20) lower band AND price < 1d EMA50 AND volume > 2.0x 20-period average.
Exit when price reverts to Donchian midline (20-period average of high/low) or ATR stoploss hits.
Uses discrete position sizing (0.30) to balance return and drawdown. Targets 20-50 trades/year per symbol.
Donchian channels provide clear breakout levels; EMA50 filters for higher-timeframe trend; volume confirms breakout strength.
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
    
    # Load 4h data for Donchian calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate ATR(14) on 4h data for stoploss
    tr1 = np.maximum(high_4h - low_4h, np.abs(high_4h - np.roll(close_4h, 1)))
    tr2 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_4h[0] - low_4h[0]  # first bar
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on 4h timeframe
    vol_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_4h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(atr_4h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 4h close for price comparison
        price_4h = close_4h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: break above upper band AND price > 1d EMA50 AND volume confirmation
            if (price_4h > donchian_upper[i] and 
                price_4h > ema50_1d_aligned[i] and 
                volume_4h[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price_4h
            # Short: break below lower band AND price < 1d EMA50 AND volume confirmation
            elif (price_4h < donchian_lower[i] and 
                  price_4h < ema50_1d_aligned[i] and 
                  volume_4h[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price_4h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midline OR ATR stoploss
                if price_4h < donchian_middle[i]:
                    exit_signal = True
                elif price_4h < entry_price - 2.5 * atr_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midline OR ATR stoploss
                if price_4h > donchian_middle[i]:
                    exit_signal = True
                elif price_4h > entry_price + 2.5 * atr_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "4H_Donchian20_1dEMA50_Volume_ATRStop"
timeframe = "4h"
leverage = 1.0