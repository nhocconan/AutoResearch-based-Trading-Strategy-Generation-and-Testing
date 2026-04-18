#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly EMA200 Trend + Daily KAMA Direction + Volume Spike
# Weekly EMA200 provides robust trend filter for both bull and bear markets.
# KAMA adapts to volatility, giving timely direction changes in choppy conditions.
# Volume spike confirms institutional participation in the move.
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_KAMA_Direction_WeeklyEMA200_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA200
    ema_200_weekly = pd.Series(df_weekly['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_200_weekly)
    
    # Calculate daily KAMA (adaptive moving average)
    # Efficiency ratio: abs(close - close[10]) / sum(abs(diff)) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of absolute changes
    # Pad arrays for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(1, np.nan), volatility])
    # Rolling sum of volatility
    vol_sum = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    # Efficiency ratio
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Volume spike: current volume > 2.0 * 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_weekly_aligned[i]) or np.isnan(kama[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_200_weekly_aligned[i]
        kama_val = kama[i]
        
        if position == 0:
            # Long: Price above weekly EMA200 AND close above KAMA AND volume spike
            if close_val > ema_val and close_val > kama_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA200 AND close below KAMA AND volume spike
            elif close_val < ema_val and close_val < kama_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price below weekly EMA200 OR close below KAMA
            if close_val < ema_val or close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price above weekly EMA200 OR close above KAMA
            if close_val > ema_val or close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals