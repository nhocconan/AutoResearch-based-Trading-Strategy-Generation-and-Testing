#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR-based breakout with 1d trend filter and volume confirmation.
# ATR breakout captures volatility expansion in both bull and bear markets.
# 1d EMA50 ensures trades align with higher timeframe trend.
# Volume confirmation filters low-conviction breakouts.
# Target: 15-30 trades per year (60-120 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) for breakout levels
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[0], tr])  # First TR is 0
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = np.zeros(len(close_1d))
    ema_multiplier = 2 / (50 + 1)
    ema_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_1d[i] = (close_1d[i] - ema_1d[i-1]) * ema_multiplier + ema_1d[i-1]
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    breakout_multiplier = 1.5  # ATR multiplier for breakout
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(avg_volume[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_val = atr[i]
        daily_ema = ema_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long breakout: price > high + ATR*mult + above daily EMA + volume
            if (price > high[i-1] + breakout_multiplier * atr_val and
                price > daily_ema and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short breakout: price < low - ATR*mult + below daily EMA + volume
            elif (price < low[i-1] - breakout_multiplier * atr_val and
                  price < daily_ema and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < daily EMA (trend change) or volatility contraction
            if price < daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > daily EMA (trend change) or volatility contraction
            if price > daily_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_ATR_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0