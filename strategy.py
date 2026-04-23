#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
Long when price breaks above 20-day high AND 1w EMA50 uptrend AND volume > 2.0x 20-period average.
Short when price breaks below 20-day low AND 1w EMA50 downtrend AND volume > 2.0x 20-period average.
Exit when price retouches 10-day EMA or ATR stoploss hit (2.5*ATR).
Uses discrete position sizing (0.30) to balance return and risk. Targets 7-25 trades/year per symbol.
1d timeframe reduces trade frequency and fee drag while capturing multi-week swings.
Works in both bull (trend continuation) and bear (mean reversion at extremes) markets.
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 1d timeframe (no further alignment needed as we're on 1d)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    
    # Load 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume average (20-period) on 1d data
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # ATR(14) for stoploss calculation (using 1d data)
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 10-day EMA for exit
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20, 14, 10)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema10_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma_1d_aligned[i]
        atr_val = atr_1d_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        ema50 = ema50_1w_aligned[i]
        ema10 = ema10_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above 20-day high AND 1w EMA50 uptrend AND volume spike
            if (price > upper and 
                close[i] > ema50 and  # Current close above EMA50 for uptrend
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.30
                position = 1
                entry_price = price
            # Short: Price breaks below 20-day low AND 1w EMA50 downtrend AND volume spike
            elif (price < lower and 
                  close[i] < ema50 and  # Current close below EMA50 for downtrend
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.30
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retouches 10-day EMA
            if position == 1 and price < ema10:
                exit_signal = True
            elif position == -1 and price > ema10:
                exit_signal = True
            
            # ATR-based stoploss: 2.5 * ATR from entry
            if position == 1 and price < entry_price - 2.5 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.5 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "1D_Donchian20_1wEMA50_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0