#!/usr/bin/env python3
"""
4h_1d_KAMA_Trend_Volume_Momentum
Hypothesis: KAMA trend direction from 1d combined with 4h volume momentum and RSI filter.
- Long when: 1d KAMA rising, 4h RSI > 55, and volume > 1.5x 20-period average
- Short when: 1d KAMA falling, 4h RSI < 45, and volume > 1.5x 20-period average
- Exit when: RSI crosses back to neutral zone (45-55) or trend reverses
Designed for fewer trades (<30/year) with momentum confirmation to work in both bull and bear markets.
"""

name = "4h_1d_KAMA_Trend_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d KAMA Trend Filter ---
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=0)[:len(change)]
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.full_like(close_1d, np.nan)
    kama[29] = close_1d[29]  # Start after enough data
    for i in range(30, len(close_1d)):
        kama[i] = kama[i-1] + sc[i-30] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_slope = np.diff(kama_1d, prepend=kama_1d[0])
    kama_1d_slope = np.append(kama_1d_slope, kama_1d_slope[-1])  # Same length
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    kama_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # --- 4h RSI (14) ---
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Momentum: 4h volume > 1.5x 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_momentum = volume_4h > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 35  # for KAMA and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(kama_slope_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d KAMA trend
        kama_rising = kama_slope_aligned[i] > 0
        kama_falling = kama_slope_aligned[i] < 0
        
        # RSI conditions
        rsi_overbought = rsi[i] > 55
        rsi_oversold = rsi[i] < 45
        rsi_neutral = (rsi[i] >= 45) & (rsi[i] <= 55)
        
        if position == 0:
            # Look for entries only with volume momentum
            if kama_rising and rsi_overbought and vol_momentum[i]:
                # Long: KAMA up + RSI > 55 + volume momentum
                signals[i] = 0.25
                position = 1
            elif kama_falling and rsi_oversold and vol_momentum[i]:
                # Short: KAMA down + RSI < 45 + volume momentum
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: RSI returns to neutral or trend reverses
            if position == 1:
                # Exit long: RSI <= 55 or KAMA turns down
                if rsi[i] <= 55 or not kama_rising:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI >= 45 or KAMA turns up
                if rsi[i] >= 45 or not kama_falling:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals