#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 1d regime filter
    # Long: Bull Power > 0 AND Bear Power < 0 AND 1d close > 1d EMA200 (bull regime)
    # Short: Bear Power < 0 AND Bull Power > 0 AND 1d close < 1d EMA200 (bear regime)
    # Uses Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure
    # bull/bear strength relative to short-term EMA. Combined with 1d trend filter
    # to avoid counter-trend trades. Discrete sizing (0.25) to minimize fee drag.
    # Target: 12-37 trades/year to stay within 6h optimal range.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for regime filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Elder Ray on 6h timeframe
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Regime filter: 1d trend direction
        bull_regime = close[i] > ema_200_1d_aligned[i]
        bear_regime = close[i] < ema_200_1d_aligned[i]
        
        # Elder Ray conditions
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0
        bearish_momentum = bear_power[i] < 0 and bull_power[i] > 0
        
        # Entry conditions: Elder Ray alignment + regime filter
        enter_long = bullish_momentum and bull_regime
        enter_short = bearish_momentum and bear_regime
        
        # Exit conditions: reverse Elder Ray signal
        exit_long = position == 1 and (bull_power[i] <= 0 or bear_power[i] >= 0)
        exit_short = position == -1 and (bull_power[i] >= 0 or bear_power[i] <= 0)
        
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_elder_ray_regime_v1"
timeframe = "6h"
leverage = 1.0