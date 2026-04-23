#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high AND close > 1w EMA50 AND volume > 1.5x 20-day average.
Short when price breaks below 20-day low AND close < 1w EMA50 AND volume > 1.5x 20-day average.
Exit when price reverts to 10-day EMA or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Donchian channels provide robust structure in both trending and ranging markets.
1w EMA50 filters for higher timeframe trend alignment, reducing counter-trend trades.
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
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit
    ema10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema10[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > 1w EMA50 AND volume confirmation
            if (price > donchian_high[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low AND close < 1w EMA50 AND volume confirmation
            elif (price < donchian_low[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to 10-day EMA or ATR stoploss
                if price < ema10[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to 10-day EMA or ATR stoploss
                if price > ema10[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0