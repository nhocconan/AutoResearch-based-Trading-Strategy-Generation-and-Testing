#!/usr/bin/env python3
# Hypothesis: 12h EMA crossover with 1d RSI filter and volume confirmation. Uses EMA(20)/EMA(50) on 12h for trend direction,
# RSI(14) on 1d > 50 for bull bias / < 50 for bear bias to avoid counter-trend trades, and volume > 1.2x 20-bar average for conviction.
# Designed to capture medium-term trends with strict filters to minimize trades and fee drag. Targets 15-25 trades/year per symbol.

name = "12h_EMA20_50_Crossover_1dRSI_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # EMA(20) and EMA(50) for trend
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.2 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # RSI(14) on 1d
    delta = pd.Series(close_1d).diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = gain / (loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    
    # Align to 12h (wait for completed 1d bar)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after EMA(50) warmup
        # Skip if missing data
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or
            np.isnan(volume_confirm[i]) or np.isnan(rsi_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: EMA20 > EMA50, RSI > 50 (bull bias), volume confirmation
        long_condition = (ema_20[i] > ema_50[i]) and (rsi_14_aligned[i] > 50) and volume_confirm[i]
        # Short conditions: EMA20 < EMA50, RSI < 50 (bear bias), volume confirmation
        short_condition = (ema_20[i] < ema_50[i]) and (rsi_14_aligned[i] < 50) and volume_confirm[i]
        
        if position == 0:
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: EMA20 < EMA50 (trend change)
            if ema_20[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: EMA20 > EMA50 (trend change)
            if ema_20[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals