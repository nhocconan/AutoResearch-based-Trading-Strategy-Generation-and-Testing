#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w trend filter and ATR-based position sizing.
Long when price breaks above 20-day high AND weekly close > weekly EMA34 AND ATR(14) < 0.03*close (low volatility).
Short when price breaks below 20-day low AND weekly close < weekly EMA34 AND ATR(14) < 0.03*close.
Exit when price touches 10-day EMA (mean reversion) or ATR expands > 0.05*close (volatility spike).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Donchian provides structure, weekly trend filter avoids counter-trend trades, ATR filter avoids chop.
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
    
    # Load 1d data for Donchian channels and EMA10 - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian channels (20-period) on 1d data
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate EMA10 on 1d data for exit
    ema10_1d = pd.Series(close_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR(14) on 1d data for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 1d timeframe (no alignment needed as primary is 1d)
    donchian_high_aligned = donchian_high
    donchian_low_aligned = donchian_low
    ema10_1d_aligned = ema10_1d
    atr14_aligned = atr14
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w data for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 1d timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema10_1d_aligned[i]) or np.isnan(atr14_aligned[i]) or
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is low (avoid choppy markets)
        vol_filter = atr14_aligned[i] < 0.03 * close[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high AND weekly uptrend AND low volatility
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema34_1w_aligned[i] and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND weekly downtrend AND low volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema34_1w_aligned[i] and vol_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price touches 10-day EMA OR volatility expands
                if (close[i] <= ema10_1d_aligned[i] or 
                    atr14_aligned[i] > 0.05 * close[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price touches 10-day EMA OR volatility expands
                if (close[i] >= ema10_1d_aligned[i] or 
                    atr14_aligned[i] > 0.05 * close[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA34_ATRFilter"
timeframe = "1d"
leverage = 1.0