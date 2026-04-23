#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike confirmation, and ATR-based stoploss.
Long when price breaks above Donchian upper band AND close > 1d EMA50 AND volume > 2.0x 20-period average.
Short when price breaks below Donchian lower band AND close < 1d EMA50 AND volume > 2.0x 20-period average.
Exit on Donchian middle band (mean) crossover or ATR trailing stop (2.5x ATR from extreme).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 19-50 trades/year per symbol.
Designed for BTC/ETH robustness in both bull and bear markets via HTF trend filter and volatility-adjusted exits.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period) on primary timeframe
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # ATR for volatility and stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0  # highest high since long entry
    short_extreme = 0.0  # lowest low since short entry
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20, 14)  # warmup for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND close > 1d EMA50 AND volume spike
            if (price > donchian_upper[i] and 
                close[i] > ema50_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                long_extreme = high[i]  # initialize extreme
            # Short: price breaks below Donchian lower AND close < 1d EMA50 AND volume spike
            elif (price < donchian_lower[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                short_extreme = low[i]  # initialize extreme
        else:
            # Update extremes for trailing stop
            if position == 1:
                long_extreme = max(long_extreme, high[i])
            else:  # position == -1
                short_extreme = min(short_extreme, low[i])
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: price crosses Donchian middle band (mean reversion)
            if position == 1 and price < donchian_middle[i]:
                exit_signal = True
            elif position == -1 and price > donchian_middle[i]:
                exit_signal = True
            
            # Secondary exit: ATR trailing stop (2.5x ATR from extreme)
            if not exit_signal:
                if position == 1 and price < long_extreme - 2.5 * atr_val:
                    exit_signal = True
                elif position == -1 and price > short_extreme + 2.5 * atr_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0