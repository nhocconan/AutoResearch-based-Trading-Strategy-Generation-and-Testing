#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h_4h_1d_camarilla_breakout_v1
# Uses 4h and 1d timeframes for signal direction: 4h trend via EMA cross, 1d momentum via price > SMA200.
# 1h timeframe for entry timing: breaks above/below prior 4h swing high/low with volume confirmation.
# Designed for low trade frequency (target: 15-37 trades/year) to minimize fee drag.
# Works in bull markets (trend continuation) and bear markets (mean reversion at swings).

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and swing levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h EMA crossover for trend direction (fast=9, slow=21)
    close_4h = df_4h['close'].values
    ema_9 = pd.Series(close_4h).ewm(span=9, min_periods=9, adjust=False).mean().values
    ema_21 = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_cross = ema_9 - ema_21  # >0 bullish, <0 bearish
    ema_cross_aligned = align_htf_to_ltf(prices, df_4h, ema_cross)
    
    # 1d price > SMA200 for bull regime filter
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    price_above_sma200 = df_1d['close'].values > sma_200_1d
    regime_aligned = align_htf_to_ltf(prices, df_1d, price_above_sma200.astype(float))
    
    # 4h swing high/low for entry levels (prior completed 4h bar)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    swing_high = np.maximum.accumulate(high_4h)  # simple proxy for resistance
    swing_low = np.minimum.accumulate(low_4h)    # simple proxy for support
    swing_high_aligned = align_htf_to_ltf(prices, df_4h, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_4h, swing_low)
    
    # Volume confirmation: volume > 1.5 * 24-period average (1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup
        # Skip if any data not ready
        if (np.isnan(ema_cross_aligned[i]) or np.isnan(regime_aligned[i]) or
            np.isnan(swing_high_aligned[i]) or np.isnan(swing_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias: bullish if 4h EMA bullish AND price above 1d SMA200
        bullish_bias = (ema_cross_aligned[i] > 0) and (regime_aligned[i] > 0.5)
        bearish_bias = (ema_cross_aligned[i] < 0) and (regime_aligned[i] <= 0.5)
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if no volume
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above 4h swing high with bullish bias
        if close[i] > swing_high_aligned[i] and bullish_bias and position != 1:
            position = 1
            signals[i] = 0.20
        # Short signal: price breaks below 4h swing low with bearish bias
        elif close[i] < swing_low_aligned[i] and bearish_bias and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit conditions: opposite breakout or bias flip
        elif (close[i] < swing_low_aligned[i] and position == 1) or \
             (close[i] > swing_high_aligned[i] and position == -1) or \
             (bullish_bias and position == -1) or \
             (bearish_bias and position == 1):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals