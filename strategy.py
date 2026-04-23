#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based stoploss.
- Long: Close > Donchian Upper(20) AND price > 1d EMA50 AND ATR(14) < 0.03 * close (low volatility)
- Short: Close < Donchian Lower(20) AND price < 1d EMA50 AND ATR(14) < 0.03 * close
- Exit: Close crosses 1d EMA50 (trend reversal) OR ATR spike > 0.05 * close (volatility expansion)
- Uses 1d HTF for EMA50 and Donchian levels (from prior completed 1d bar)
- Designed for low trade frequency (15-40/year) to minimize fee drag
- Works in bull (buy breakouts above upper band) and bear (sell breakdowns below lower band)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR(14) for volatility filter and stop condition
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]  # first bar
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Donchian(20) levels (HTF = 1d)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA, 20 for Donchian, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_upper_aligned[i]) or
            np.isnan(donchian_lower_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filters
        low_vol = atr[i] < 0.03 * close[i]   # Low volatility environment
        vol_expansion = atr[i] > 0.05 * close[i]  # Volatility spike (exit condition)
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > donchian_upper_aligned[i-1]   # Close above prior upper band
        breakout_down = close[i] < donchian_lower_aligned[i-1] # Close below prior lower band
        
        if position == 0:
            # Long: Donchian upper breakout AND price > 1d EMA50 AND low volatility
            if breakout_up and close[i] > ema_50_1d_aligned[i] and low_vol:
                signals[i] = 0.25
                position = 1
            # Short: Donchian lower breakout AND price < 1d EMA50 AND low volatility
            elif breakout_down and close[i] < ema_50_1d_aligned[i] and low_vol:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price < 1d EMA50 (trend reversal) OR volatility expansion
            if close[i] < ema_50_1d_aligned[i] or vol_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price > 1d EMA50 (trend reversal) OR volatility expansion
            if close[i] > ema_50_1d_aligned[i] or vol_expansion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_ATR_VolFilter"
timeframe = "4h"
leverage = 1.0