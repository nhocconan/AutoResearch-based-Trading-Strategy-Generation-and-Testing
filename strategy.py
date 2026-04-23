#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume spike filter.
Long when price breaks above 20-day high AND close > 1w EMA50 AND volume > 2.0x 20-day average.
Short when price breaks below 20-day low AND close < 1w EMA50 AND volume > 2.0x 20-day average.
Exit when price crosses 10-day EMA in opposite direction or ATR stoploss (2.5x ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Designed for BTC/ETH performance in both bull and bear markets via trend filter and volume confirmation.
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
    
    # Load 1d data for price action - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR(14) on 1d data for stoploss
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as primary timeframe is 1d)
    donchian_high_aligned = donchian_high  # already 1d
    donchian_low_aligned = donchian_low    # already 1d
    ema10_1d_aligned = ema10_1d            # already 1d
    atr_1d_aligned = atr_1d                # already 1d
    vol_ma_aligned = vol_ma                # already 1d
    
    # Align 1w EMA50 to 1d timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema10_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND close > 1w EMA50 AND volume spike
            if (price > donchian_high_aligned[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian low AND close < 1w EMA50 AND volume spike
            elif (price < donchian_low_aligned[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 10-day EMA or ATR stoploss
                if price < ema10_1d_aligned[i]:
                    exit_signal = True
                elif price < entry_price - 2.5 * atr_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 10-day EMA or ATR stoploss
                if price > ema10_1d_aligned[i]:
                    exit_signal = True
                elif price > entry_price + 2.5 * atr_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeSpike_EMA10Exit"
timeframe = "1d"
leverage = 1.0