#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Squeeze_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for BB and Keltner calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Squeeze Momentum: Bollinger Bands within Keltner Channel
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Keltner Channel (20, 1.5)
    atr_1d = pd.Series(high_1d - low_1d).rolling(window=20, min_periods=20).mean().values
    kc_upper = sma_20 + 1.5 * atr_1d
    kc_lower = sma_20 - 1.5 * atr_1d
    
    # Squeeze condition: BB inside KC
    squeeze_on = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    squeeze_off = ~squeeze_on
    
    # Momentum: close - SMA(20)
    momentum = close_1d - sma_20
    
    # Align to 4h timeframe
    squeeze_on_aligned = align_htf_to_ltf(prices, df_1d, squeeze_on)
    squeeze_off_aligned = align_htf_to_ltf(prices, df_1d, squeeze_off)
    momentum_aligned = align_htf_to_ltf(prices, df_1d, momentum)
    
    # 4h trend filter: EMA(50)
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume filter: volume > 1.5 * average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: squeeze off, momentum > 0, price above EMA50, volume confirmation
            long_cond = squeeze_off_aligned[i] and (momentum_aligned[i] > 0) and (close[i] > ema_50[i]) and vol_filter[i]
            # Short: squeeze off, momentum < 0, price below EMA50, volume confirmation
            short_cond = squeeze_off_aligned[i] and (momentum_aligned[i] < 0) and (close[i] < ema_50[i]) and vol_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: squeeze on or momentum <= 0 or price below EMA50
            if squeeze_on_aligned[i] or (momentum_aligned[i] <= 0) or (close[i] <= ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: squeeze on or momentum >= 0 or price above EMA50
            if squeeze_on_aligned[i] or (momentum_aligned[i] >= 0) or (close[i] >= ema_50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Squeeze Breakout Strategy
# Uses Bollinger Bands within Keltner Channel to identify low volatility squeezes
# Breakout occurs when price exits squeeze with momentum and volume confirmation
# Trend filter ensures trades align with higher timeframe direction
# Works in both bull and bear markets by capturing volatility expansion phases
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag
# Squeeze indicates consolidation; breakout with volume signals new trend start
# EMA50 filter avoids counter-trend trades during strong trends
# Volume confirmation reduces false breakouts in low liquidity periods