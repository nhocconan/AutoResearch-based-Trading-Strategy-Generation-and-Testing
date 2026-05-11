#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_Volume_Momentum_v2
Hypothesis: KAMA direction (trend) from 1d timeframe combined with 4h RSI momentum and volume confirmation.
- Long when: 1d KAMA rising (bullish trend), 4h RSI > 50 (momentum), and volume > 20-period average
- Short when: 1d KAMA falling (bearish trend), 4h RSI < 50 (momentum), and volume > 20-period average
- Exit when: RSI crosses back to neutral (50) or trend changes
Designed to work in both bull (trend following) and bear (mean reversion via RSI) regimes.
Targets ~25 trades/year (100 over 4 years) to minimize fee drag.
Uses 1d trend filter to avoid counter-trend trades.
"""

name = "4h_1d_KAMA_Trend_Volume_Momentum_v2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast=2, slow=30):
    """Kaufman's Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    
    # Efficiency Ratio
    er = np.zeros_like(close)
    for i in range(len(close)):
        if i >= er_period:
            price_change = np.abs(close[i] - close[i-er_period])
            sum_volatility = np.sum(volatility[i-er_period+1:i+1])
            if sum_volatility > 0:
                er[i] = price_change / sum_volatility
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # KAMA calculation
    kama_vals = np.zeros_like(close)
    kama_vals[0] = close[0]
    for i in range(1, len(close)):
        kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
    
    return kama_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for KAMA calculation
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: KAMA ---
    close_1d = df_1d['close'].values
    kama_1d = kama(close_1d, er_period=10, fast=2, slow=30)
    kama_1d_prev = np.roll(kama_1d, 1)
    kama_1d_prev[0] = kama_1d[0]
    kama_rising = kama_1d > kama_1d_prev
    kama_falling = kama_1d < kama_1d_prev
    
    # Align KAMA direction to 4h timeframe
    kama_rising_4h = align_htf_to_ltf(prices, df_1d, kama_rising)
    kama_falling_4h = align_htf_to_ltf(prices, df_1d, kama_falling)
    
    # --- 4h RSI Momentum (14-period) ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (max of KAMA and RSI periods)
    start_idx = max(30, 20)  # KAMA needs ~30, RSI needs 14, VOL needs 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_rising_4h[i]) or np.isnan(kama_falling_4h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries with volume confirmation
            if kama_rising_4h[i] and rsi[i] > 50 and vol_ok:
                # Long: bullish KAMA + bullish RSI momentum + volume
                signals[i] = 0.25
                position = 1
            elif kama_falling_4h[i] and rsi[i] < 50 and vol_ok:
                # Short: bearish KAMA + bearish RSI momentum + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI returns to neutral (50) or trend changes
                if rsi[i] <= 50 or not kama_rising_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI returns to neutral (50) or trend changes
                if rsi[i] >= 50 or not kama_falling_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals