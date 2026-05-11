#!/usr/bin/env python3
# 4h_ThreeBarReversal_1dTrend_Volume_Confirm
# Hypothesis: Three-bar reversal pattern (three consecutive closes in opposite direction) 
# combined with 1-day EMA50 trend filter and volume confirmation. The pattern captures
# momentum exhaustion and potential reversals. Works in both bull and bear markets
# by requiring trend alignment (only take reversals against the trend for mean reversion,
# or with the trend for continuation - but here we use counter-trend reversals in trending markets).
# Targets 20-30 trades/year to minimize fee drag.

name = "4h_ThreeBarReversal_1dTrend_Volume_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # === 1d Data (loaded ONCE) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 1d EMA50 Trend Filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Three-bar reversal pattern ===
    # Bearish reversal: three consecutive higher closes (uptrend exhaustion)
    bullish_reversal = np.zeros(n, dtype=bool)
    bearish_reversal = np.zeros(n, dtype=bool)
    
    for i in range(2, n):
        # Three consecutive higher closes = potential bearish reversal setup
        if (close[i-2] < close[i-1] < close[i]):
            bearish_reversal[i] = True
        # Three consecutive lower closes = potential bullish reversal setup
        if (close[i-2] > close[i-1] > close[i]):
            bullish_reversal[i] = True
    
    # === Signal Parameters ===
    position_size = 0.25  # 25% of capital per trade
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    holding_bars = 0
    
    # Start after warmup (covers EMA50)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_4h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                holding_bars = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Three-bar bullish reversal + below 1d EMA50 (mean reversion in downtrend) + volume
            if (bullish_reversal[i] and 
                close[i] < ema50_1d_4h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
                holding_bars = 0
            # Short: Three-bar bearish reversal + above 1d EMA50 (mean reversion in uptrend) + volume
            elif (bearish_reversal[i] and 
                  close[i] > ema50_1d_4h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
                holding_bars = 0
        else:
            # Enforce minimum holding period (8 bars)
            holding_bars += 1
            if holding_bars < 8:
                signals[i] = position_size if position == 1 else -position_size
                continue
            
            # Exit: Three-bar reversal in opposite direction
            if position == 1:
                if bearish_reversal[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if bullish_reversal[i]:
                    signals[i] = 0.0
                    position = 0
                    holding_bars = 0
                else:
                    signals[i] = -position_size
    
    return signals