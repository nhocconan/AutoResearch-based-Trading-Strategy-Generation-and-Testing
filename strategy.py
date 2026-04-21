#!/usr/bin/env python3
"""
4h_1d_RSI_MeanReversion_VolumeFilter
Hypothesis: Daily RSI extreme readings (overbought/oversold) combined with volume exhaustion and 4h mean reversion provide edge in both bull and bear markets.
Uses daily RSI for regime, 4h price action for entry, and volume to filter low-conviction moves. Targets 20-40 trades/year with controlled risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for RSI
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    close_daily = df_daily['close'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_daily, prepend=close_daily[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs))
    rsi_daily = rsi_daily.values
    
    # Align daily RSI to 4h timeframe (no extra delay needed for RSI)
    rsi_daily_aligned = align_htf_to_ltf(prices, df_daily, rsi_daily)
    
    # 4h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h RSI for entry timing (optional, can use price action instead)
    delta_4h = np.diff(close, prepend=close[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in daily RSI
        if np.isnan(rsi_daily_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_daily_val = rsi_daily_aligned[i]
        rsi_4h_val = rsi_4h[i]
        vol_current = volume[i]
        
        # Volume filter: current volume < 0.7x 20-period average (volume exhaustion for mean reversion)
        vol_ma = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        vol_exhausted = vol_current < 0.7 * vol_ma
        
        if position == 0:
            # Long setup: daily RSI oversold (<30) + 4h RSI recovering from oversold + volume exhaustion
            if rsi_daily_val < 30 and rsi_4h_val < 30 and rsi_4h_val > rsi_4h[i-1] and vol_exhausted:
                signals[i] = 0.25
                position = 1
            # Short setup: daily RSI overbought (>70) + 4h RSI declining from overbought + volume exhaustion
            elif rsi_daily_val > 70 and rsi_4h_val > 70 and rsi_4h_val < rsi_4h[i-1] and vol_exhausted:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: daily RSI returns to neutral (50) or 4h RSI overbought
            if rsi_daily_val >= 50 or rsi_4h_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: daily RSI returns to neutral (50) or 4h RSI oversold
            if rsi_daily_val <= 50 or rsi_4h_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_RSI_MeanReversion_VolumeFilter"
timeframe = "4h"
leverage = 1.0