#!/usr/bin/env python3
# 12h_WeeklyBullBear_Signal
# Hypothesis: Uses weekly EMA cross for trend direction (bull/bear) and daily RSI for mean-reversion entries.
# In bull trend (weekly close > weekly EMA50), go long when daily RSI < 30.
# In bear trend (weekly close < weekly EMA50), go short when daily RSI > 70.
# Filters entries with daily volume > 1.5x 20-day average to avoid low-liquidity false signals.
# Designed to work in both bull and bear markets by following the weekly trend while buying dips/selling rallies.
# Target: 15-25 trades/year per symbol with low turnover to minimize fee drag.

name = "12h_WeeklyBullBear_Signal"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema_50 = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        ema_50[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            ema_50[i] = (close_1w[i] * 2 + ema_50[i-1] * 49) / 51
    
    # Get daily data for RSI and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily volume 20-day average
    vol_ma = np.full_like(volume_1d, np.nan)
    if len(volume_1d) >= 20:
        vol_ma[19] = np.mean(volume_1d[0:20])
        for i in range(20, len(volume_1d)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume_1d[i]) / 20
    
    volume_ratio = np.full_like(volume_1d, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume_1d[valid_vol] / vol_ma[valid_vol]
    
    # Align weekly EMA50, daily RSI, and daily volume ratio to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    volume_ratio_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_ratio_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: weekly close vs weekly EMA50
        weekly_close = close_1d[-1] if hasattr(close_1d, '__getitem__') else close_1d[-1]  # placeholder for logic
        # Actually, we need the weekly close value aligned - but we don't have it directly.
        # Instead, we'll use the condition: bull trend when current 12h close > aligned weekly EMA50
        # This approximates the weekly trend direction.
        bull_trend = close[i] > ema_50_aligned[i]
        bear_trend = close[i] < ema_50_aligned[i]
        
        # Volume filter
        vol_filter = volume_ratio_aligned[i] > 1.5
        
        if position == 0:
            # Enter long: bull trend AND RSI oversold AND volume confirmation
            if bull_trend and rsi_aligned[i] < 30 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bear trend AND RSI overbought AND volume confirmation
            elif bear_trend and rsi_aligned[i] > 70 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI overbought OR trend turns bear
            if rsi_aligned[i] > 70 or not bull_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold OR trend turns bull
            if rsi_aligned[i] < 30 or not bear_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals