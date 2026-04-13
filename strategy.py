#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d regime filter
    # Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
    # Long: Bull Power > 0 AND Bear Power < 0 AND 1d close > 1d EMA50 (bull regime)
    # Short: Bear Power > 0 AND Bull Power < 0 AND 1d close < 1d EMA50 (bear regime)
    # Uses discrete sizing (0.25) to minimize fee drag
    # Target: 12-37 trades/year to stay within 6h optimal range
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time']
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray components on 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d trend direction
        bull_regime = close[i] > ema_50_1d_aligned[i]  # Price above 1d EMA50 = bull regime
        bear_regime = close[i] < ema_50_1d_aligned[i]  # Price below 1d EMA50 = bear regime
        
        # Elder Ray signals: look for divergence between price and power
        long_signal = (bull_power[i] > 0) and (bear_power[i] < 0) and bull_regime
        short_signal = (bear_power[i] > 0) and (bull_power[i] < 0) and bear_regime
        
        # Exit conditions: reverse signal or power divergence failure
        exit_long = position == 1 and ((bull_power[i] <= 0) or (bear_power[i] >= 0) or not bull_regime)
        exit_short = position == -1 and ((bear_power[i] <= 0) or (bull_power[i] >= 0) or not bear_regime)
        
        # Execute signals
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_v2"
timeframe = "6h"
leverage = 1.0