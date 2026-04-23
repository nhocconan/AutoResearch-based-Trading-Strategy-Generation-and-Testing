#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper band AND close > 1w EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below lower band AND close < 1w EMA50 AND volume > 1.5x 20-period average.
Exit when price reverts to the middle band (20-period SMA) or ATR-based stoploss hits.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-30 trades/year per symbol.
Donchian channels provide clear breakout levels that work in trending markets, while 1w EMA50 filters for higher-timeframe trend alignment.
Volume confirmation ensures institutional participation. This combination should work in both bull and bear markets by avoiding counter-trend breakouts.
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
    
    # Load 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels for 1d timeframe (using previous bar's data)
    # Upper band = 20-period high, Lower band = 20-period low, Middle band = 20-period SMA
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d Donchian levels to 1d timeframe (no additional delay needed)
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    middle_1d_aligned = align_htf_to_ltf(prices, df_1d, middle_1d)
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
    tr2 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(middle_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper band AND close > 1w EMA50 AND volume confirmation
            if (price > upper_1d_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower band AND close < 1w EMA50 AND volume confirmation
            elif (price < lower_1d_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to middle band or ATR stoploss
                if price <= middle_1d_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.0 * atr_1d[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to middle band or ATR stoploss
                if price >= middle_1d_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.0 * atr_1d[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0