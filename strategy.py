#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x average.
Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x average.
Exit when price reverses to opposite Donchian level or ATR-based stoploss.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-30 trades/year per symbol.
Donchian channels provide clear breakout levels, EMA50 filters for higher timeframe trend,
and volume confirmation reduces false breakouts.
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
    
    # Load 12h data for Donchian calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian(20) on 12h data
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) on 12h data for stoploss
    tr1 = np.maximum(high_12h - low_12h, np.abs(high_12h - np.roll(close_12h, 1)))
    tr2 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_12h[0] - low_12h[0]  # first bar
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 12h timeframe
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(ema50_1w_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        # Use 12h close for price comparison
        price_12h = close_12h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > 1w EMA50 AND volume confirmation
            if (price_12h > donchian_high[i] and 
                price_12h > ema50_1w_aligned[i] and 
                volume_12h[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price_12h
            # Short: price breaks below Donchian low AND price < 1w EMA50 AND volume confirmation
            elif (price_12h < donchian_low[i] and 
                  price_12h < ema50_1w_aligned[i] and 
                  volume_12h[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price_12h
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverses to Donchian low OR ATR stoploss
                if price_12h < donchian_low[i]:
                    exit_signal = True
                elif price_12h < entry_price - 2.5 * atr_12h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverses to Donchian high OR ATR stoploss
                if price_12h > donchian_high[i]:
                    exit_signal = True
                elif price_12h > entry_price + 2.5 * atr_12h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1wEMA50_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0