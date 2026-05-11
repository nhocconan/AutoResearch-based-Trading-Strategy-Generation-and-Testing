# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
4h_Range_Trend_Filter_v1
Hypothesis: Uses 1-day range (high-low) and ATR to identify volatility regime.
In low volatility (ATR < 20-period ATR mean), trade mean reversion at Bollinger Bands.
In high volatility (ATR > 20-period ATR mean), trade breakouts of Donchian channels.
Uses 1-day trend (EMA50) to filter trades: only long when price > EMA50, short when price < EMA50.
Designed for low frequency (20-40 trades/year) to work in both bull (breakouts) and bear (mean reversion) markets.
"""

name = "4h_Range_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1-day EMA50 for trend filter ---
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # --- Volatility Regime: ATR(14) vs its 20-period mean ---
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    low_vol = atr < atr_ma  # Low volatility regime
    
    # --- Bollinger Bands (20,2) for mean reversion ---
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    
    # --- Donchian Channel (20) for breakouts ---
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or
            np.isnan(std_20[i]) or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime based on volatility
        is_low_vol = low_vol[i]
        
        # Entry signals
        long_signal = False
        short_signal = False
        
        if is_low_vol:
            # Low volatility: mean reversion at Bollinger Bands
            long_signal = (close[i] < lower_bb[i]) and (close[i] > ema_50_1d_aligned[i])
            short_signal = (close[i] > upper_bb[i]) and (close[i] < ema_50_1d_aligned[i])
        else:
            # High volatility: breakout of Donchian channels
            long_signal = (high[i] > highest_high[i-1]) and (close[i] > ema_50_1d_aligned[i])
            short_signal = (low[i] < lowest_low[i-1]) and (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        else:
            # Exit conditions: opposite signal or volatility regime change
            if position == 1:
                exit_signal = short_signal or (not is_low_vol and low_vol[i])  # Exit on opposite signal or shift to low vol
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                exit_signal = long_signal or (not is_low_vol and low_vol[i])  # Exit on opposite signal or shift to low vol
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals