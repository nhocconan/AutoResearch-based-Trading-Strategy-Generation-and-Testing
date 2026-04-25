#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_ChopFilter
Hypothesis: 4h strategy using Kaufman Adaptive Moving Average (KAMA) for trend direction,
combined with RSI(14) for momentum and Choppiness Index for regime filtering.
Long when KAMA slope positive, RSI > 50, and CHOP < 38.2 (trending regime).
Short when KAMA slope negative, RSI < 50, and CHOP < 38.2.
Exit on opposite RSI cross (50) or CHOP > 61.8 (range regime).
Uses discrete sizing (0.25) to minimize fee churn. Target: 30-60 trades/year.
Works in bull via KAMA trend following, in bear via RSI mean reversion in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation (HTF regime filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr_1d = np.maximum(high_1d - low_1d,
                       np.absolute(high_1d - np.roll(close_1d, 1)),
                       np.absolute(low_1d - np.roll(close_1d, 1)))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar
    
    # Calculate ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over last 14 periods for 1d
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Calculate Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (max_high - min_low)) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    range_14 = max_high_14 - min_low_14
    chop_1d = np.full_like(close_1d, np.nan)
    valid_range = (range_14 > 0) & (~np.isnan(sum_atr_14))
    chop_1d[valid_range] = 100 * np.log10(sum_atr_14[valid_range] / range_14[valid_range]) / np.log10(14)
    
    # Align Choppiness Index to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Get 4h data for KAMA calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate Kaufman Adaptive Moving Average (KAMA) with ER=10, slow=30, fast=2
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change_10 = np.absolute(np.subtract(close_4h[10:], close_4h[:-10]))
    volatility_10 = np.sum(np.absolute(np.diff(close_4h.reshape(-1,1), axis=0).reshape(-1)), axis=0) if len(close_4h) > 1 else np.array([])
    # Simplified volatility calculation for efficiency
    volatility_10 = np.array([np.sum(np.absolute(np.diff(close_4h[max(0,i-9):i+1]))) for i in range(len(close_4h))])
    er_10 = np.divide(change_10, volatility_10[9:], out=np.full_like(change_10, np.nan), where=volatility_10[9:]!=0)
    er_10 = np.concatenate([np.full(10, np.nan), er_10])
    
    # Smoothing constants: sc = [ER * (fastest - slowest) + slowest]^2
    fastest = 2.0 / (2 + 1)  # 2-period EMA
    slowest = 2.0 / (30 + 1)  # 30-period EMA
    sc_10 = (er_10 * (fastest - slowest) + slowest) ** 2
    sc_10 = np.nan_to_num(sc_10, nan=slowest**2)
    
    # Calculate KAMA
    kama_4h = np.full_like(close_4h, np.nan)
    kama_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama_4h[i] = kama_4h[i-1] + sc_10[i] * (close_4h[i] - kama_4h[i-1])
    
    # Align KAMA to original timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    
    # Calculate KAMA slope (direction): positive if current > previous
    kama_slope = np.diff(kama_4h_aligned, prepend=kama_4h_aligned[0])
    
    # Get 4h data for RSI calculation
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average (less strict to avoid overtrading)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(kama_slope[i]) or 
            np.isnan(rsi_4h_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, trending regime (CHOP < 38.2), volume spike
            long_signal = (kama_slope[i] > 0) and (rsi_4h_aligned[i] > 50) and \
                          (chop_1d_aligned[i] < 38.2) and vol_spike[i]
            # Short: KAMA down, RSI < 50, trending regime (CHOP < 38.2), volume spike
            short_signal = (kama_slope[i] < 0) and (rsi_4h_aligned[i] < 50) and \
                           (chop_1d_aligned[i] < 38.2) and vol_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: RSI < 50 or choppy regime (CHOP > 61.8) or KAMA slope turns down
            exit_signal = (rsi_4h_aligned[i] < 50) or (chop_1d_aligned[i] > 61.8) or (kama_slope[i] <= 0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: RSI > 50 or choppy regime (CHOP > 61.8) or KAMA slope turns up
            exit_signal = (rsi_4h_aligned[i] > 50) or (chop_1d_aligned[i] > 61.8) or (kama_slope[i] >= 0)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_KAMA_Direction_RSI_ChopFilter"
timeframe = "4h"
leverage = 1.0