#!/usr/bin/env python3
"""
4h_12h_KAMA_RSI_Volume_Regime_v1
Hypothesis: Use KAMA trend from 4h and RSI extremes from 12h, filtered by volume spike and chop regime.
Only trade when price is in strong trend (KAMA direction) and RSI is overextended (>70 or <30) on 12h.
Volume must be >1.5x 20-period average to confirm momentum.
Chop regime filter: avoid trading when chop > 61.8 (range-bound).
Works in bull markets (continuation) and bear markets (mean reversion from extremes).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_KAMA_RSI_Volume_Regime_v1"
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
    
    # === KAMA TREND ON 4H ===
    close_series = pd.Series(close)
    # Efficiency Ratio
    change = abs(close - np.roll(close, 10))
    change[:10] = np.nan
    volatility = abs(np.diff(close, prepend=np.nan))
    volatility_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    er = change / volatility_sum
    er = np.nan_to_num(er, nan=0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # === RSI ON 12H ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    rsi_period = 14
    delta = np.diff(df_12h['close'].values, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_period, min_periods=rsi_period).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 0, rs)
    rsi = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # === VOLUME FILTER ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === CHOP REGIME FILTER (using 12h data) ===
    # Chop = 100 * log10(sum(abs(close - open), 14) / (max(high,14) - min(low,14))) / log10(14)
    chop_period = 14
    true_range = np.maximum(df_12h['high'].values - df_12h['low'].values,
                           np.maximum(abs(df_12h['high'].values - np.roll(df_12h['close'].values, 1)),
                                     abs(df_12h['low'].values - np.roll(df_12h['close'].values, 1))))
    tr_sum = pd.Series(true_range).rolling(window=chop_period, min_periods=chop_period).sum().values
    max_high = pd.Series(df_12h['high'].values).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(df_12h['low'].values).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(tr_sum / (max_high - min_low)) / np.log10(chop_period)
    chop = np.nan_to_num(chop, nan=100)
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(kama[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_12h_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend direction from KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI extremes on 12h
        rsi_overbought = rsi_12h_aligned[i] > 70
        rsi_oversold = rsi_12h_aligned[i] < 30
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Chop regime filter: only trade when NOT in strong range (chop <= 61.8)
        not_choppy = chop_12h_aligned[i] <= 61.8
        
        # Long: price above KAMA, RSI oversold (mean reversion), strong volume, not choppy
        long_signal = (price_above_kama and rsi_oversold and 
                      strong_volume and not_choppy)
        
        # Short: price below KAMA, RSI overbought (mean reversion), strong volume, not choppy
        short_signal = (price_below_kama and rsi_overbought and 
                       strong_volume and not_choppy)
        
        # Exit: opposite condition or volatility drop
        exit_long = (position == 1 and 
                    (price_below_kama or not strong_volume))
        exit_short = (position == -1 and 
                     (price_above_kama or not strong_volume))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals