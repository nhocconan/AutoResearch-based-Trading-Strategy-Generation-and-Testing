#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1d EMA50 uptrend AND volume > 1.8x 20-period average.
Short when price breaks below Donchian lower band AND 1d EMA50 downtrend AND volume > 1.8x 20-period average.
Exit when price retraces to Donchian midpoint OR ATR stoploss hit (2.0*ATR).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-30 trades/year per symbol.
Donchian channels provide robust trend-following structure, while 1d EMA50 ensures alignment with higher-timeframe trend.
Volume confirmation filters weak breakouts. Designed to work in both trending and ranging markets.
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
    
    # Calculate Donchian channels (20-period) on primary timeframe
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 20, 20, 14, 50)  # Ensure warmup for Donchian(20), Vol MA(20), ATR(14), EMA50(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        midpoint = donchian_mid[i]
        ema50 = ema50_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper band AND 1d EMA50 uptrend AND volume spike
            if (price > upper and 
                close[i] > ema50 and  # Current close above EMA50 for uptrend
                volume[i] > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Price breaks below Donchian lower band AND 1d EMA50 downtrend AND volume spike
            elif (price < lower and 
                  close[i] < ema50 and  # Current close below EMA50 for downtrend
                  volume[i] > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Donchian midpoint
            if position == 1 and price <= midpoint:
                exit_signal = True
            elif position == -1 and price >= midpoint:
                exit_signal = True
            
            # ATR-based stoploss: 2.0 * ATR from entry
            if position == 1 and price < entry_price - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > entry_price + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0