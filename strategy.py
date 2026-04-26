#!/usr/bin/env python3
"""
4h_KAMA_Trend_With_Volume_Regime_Filter
Hypothesis: On 4h timeframe, KAMA (adaptive trend) identifies strong trends while volume regime filter (current volume > 1.5x 50-bar average) ensures institutional participation. Long when KAMA rising + volume regime; short when KAMA falling + volume regime. Uses ATR-based stoploss (2.0x ATR) and discrete position sizing (0.25) to minimize fee drag. Designed to work in both bull and bear markets by adapting to trend strength and avoiding choppy regimes.
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
    
    # Get 1d data for HTF trend filter (more stable than lower timeframes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for HTF trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # KAMA calculation (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period sum of absolute changes
    # Fix array lengths
    change = np.concatenate([[np.nan] * 10, change])
    volatility = np.concatenate([[np.nan] * 10, volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start after 10 periods
    for i in range(10, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume regime: current volume > 1.5x 50-period average (institutional participation)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_regime = volume > 1.5 * vol_ma
    
    # ATR (14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 50, 14)  # EMA50, KAMA, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(kama[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        kama_val = kama[i]
        kama_prev = kama[i-1] if i > 0 else kama_val
        vol_regime_val = volume_regime[i]
        close_val = close[i]
        atr_val = atr[i]
        
        # KAMA direction: rising if current > previous
        kama_rising = kama_val > kama_prev
        kama_falling = kama_val < kama_prev
        
        if position == 0:
            # Long: KAMA rising + price above HTF EMA50 + volume regime
            long_signal = kama_rising and (close_val > ema_50_val) and vol_regime_val
            # Short: KAMA falling + price below HTF EMA50 + volume regime
            short_signal = kama_falling and (close_val < ema_50_val) and vol_regime_val
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. KAMA falling (trend weakening)
            if kama_falling:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. ATR-based stoploss: price drops below entry - 2.0 * ATR
            elif close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. KAMA rising (trend weakening)
            if kama_rising:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. ATR-based stoploss: price rises above entry + 2.0 * ATR
            elif close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "4h_KAMA_Trend_With_Volume_Regime_Filter"
timeframe = "4h"
leverage = 1.0