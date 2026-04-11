#!/usr/bin/env python3
# 12h_1d_ema_rsi_momentum_v1
# Strategy: 12h EMA crossover with RSI momentum filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: EMA crossovers capture medium-term trends. RSI > 50 confirms bullish momentum,
# RSI < 50 confirms bearish momentum. Volume > 1.5x 20-period average confirms institutional
# participation. Designed for low trade frequency (~20-40/year) to minimize fee drag.
# Works in bull markets via trend continuation and bear markets via short signals during
# distribution phases.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_rsi_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h EMA25 and EMA50 for crossover
    ema_25 = pd.Series(close).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(ema_25[i]) or np.isnan(ema_50[i]) or np.isnan(rsi[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # EMA crossover signals
        ema_cross_up = ema_25[i] > ema_50[i] and ema_25[i-1] <= ema_50[i-1]
        ema_cross_down = ema_25[i] < ema_50[i] and ema_25[i-1] >= ema_50[i-1]
        
        # RSI momentum filter: >50 for bullish, <50 for bearish
        rsi_bullish = rsi[i] > 50
        rsi_bearish = rsi[i] < 50
        
        # Entry conditions
        # Long: EMA bullish crossover AND RSI bullish AND volume confirmation
        if ema_cross_up and rsi_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: EMA bearish crossover AND RSI bearish AND volume confirmation
        elif ema_cross_down and rsi_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite EMA crossover (trend change)
        elif position == 1 and ema_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and ema_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals