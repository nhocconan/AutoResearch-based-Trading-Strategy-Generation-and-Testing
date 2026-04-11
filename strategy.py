#!/usr/bin/env python3
# 6h_12h_1d_triple_barrier_v1
# Strategy: Triple barrier system using 12h/1d structure with 6h entries
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Combines 12h Donchian channel breakouts with 1d volume confirmation
# and 6h momentum filters to capture multi-timeframe trends while minimizing false breakouts.
# Works in bull markets via breakout continuation and bear markets via breakdown continuation.
# Designed for low trade frequency (~15-25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_triple_barrier_v1"
timeframe = "6h"
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
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    # 6h momentum: RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]) or \
           np.isnan(rsi[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # 12h Donchian breakout/breakdown signals
        breakout_up = close[i] > donch_high_12h_aligned[i-1]  # Break above 12h Donchian high
        breakdown_down = close[i] < donch_low_12h_aligned[i-1]  # Break below 12h Donchian low
        
        # 6h momentum filter: RSI > 55 for bullish, RSI < 45 for bearish
        rsi_bullish = rsi[i] > 55
        rsi_bearish = rsi[i] < 45
        
        # Entry conditions
        # Long: 12h Donchian breakout AND volume confirmation AND 6h bullish momentum
        if breakout_up and vol_confirm and rsi_bullish and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: 12h Donchian breakdown AND volume confirmation AND 6h bearish momentum
        elif breakdown_down and vol_confirm and rsi_bearish and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite 12h Donchian break/breakdown (trend change)
        elif position == 1 and breakdown_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals