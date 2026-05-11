#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_20EMA_Filter
# Hypothesis: Uses 1d Elder Ray (Bull/Bear Power) to capture institutional buying/selling pressure
# combined with 20-period EMA on 6h for trend alignment. In bull markets (1d EMA50 > EMA200),
# long when Bull Power > 0 and price > EMA20; in bear markets (1d EMA50 < EMA200), short when
# Bear Power < 0 and price < EMA20. Volume confirmation reduces false signals.
# Designed for low turnover (target 15-25 trades/year) with strong trend persistence in 6h timeframe.

name = "6h_ElderRay_BullBearPower_1dTrend_20EMA_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === 1d Data for Elder Ray and Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA13 (for Elder Ray) and EMA50/EMA200 (trend filter)
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align 1d indicators to 6h
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_6h = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === 6h EMA20 for Entry Timing ===
    ema20_6h = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5  # Require 1.5x average volume
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of EMA200 and EMA20)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(ema50_1d_6h[i]) or np.isnan(ema200_1d_6h[i]) or 
            np.isnan(ema20_6h[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine market regime from 1d EMA50 vs EMA200
            bull_market = ema50_1d_6h[i] > ema200_1d_6h[i]
            bear_market = ema50_1d_6h[i] < ema200_1d_6h[i]
            
            # Long: Bull market + Bull Power > 0 + price > EMA20 + volume spike
            if (bull_market and 
                bull_power_6h[i] > 0 and 
                close[i] > ema20_6h[i] and 
                volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Bear market + Bear Power < 0 + price < EMA20 + volume spike
            elif (bear_market and 
                  bear_power_6h[i] < 0 and 
                  close[i] < ema20_6h[i] and 
                  volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Reverse of entry conditions or EMA crossover
            if position == 1:
                # Exit long: Bear power turns negative OR price < EMA20
                if bear_power_6h[i] < 0 or close[i] < ema20_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: Bull power turns positive OR price > EMA20
                if bull_power_6h[i] > 0 or close[i] > ema20_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals