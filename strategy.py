#!/usr/bin/env python3
# 4h_1d_rvi_volume_v1
# Strategy: 4h Relative Vigor Index (RVI) with 1d trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: RVI measures trend strength by comparing closing-open range to trading range.
# Combined with 1d EMA trend filter and volume confirmation to capture strong trends.
# Designed for low trade frequency (<50/year) to avoid fee drag in BTC/ETH markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rvi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate RVI (Relative Vigor Index) - 10 period
    # Numerator: (close - open) + 2*(close_prev - open_prev) + 2*(close_prev2 - open_prev2) + (close_prev3 - open_prev3)
    # Denominator: (high - low) + 2*(high_prev - low_prev) + 2*(high_prev2 - low_prev2) + (high_prev3 - low_prev3)
    # RVI = SMA(numerator, 4) / SMA(denominator, 4)
    
    num = (close - open_price) + \
          2 * np.roll(close - open_price, 1) + \
          2 * np.roll(close - open_price, 2) + \
          np.roll(close - open_price, 3)
    den = (high - low) + \
          2 * np.roll(high - low, 1) + \
          2 * np.roll(high - low, 2) + \
          np.roll(high - low, 3)
    
    # Handle first 3 values where roll creates invalid data
    num[:3] = np.nan
    den[:3] = np.nan
    
    # Smoothed RVI
    rvi_num = pd.Series(num).rolling(window=4, min_periods=4).mean().values
    rvi_den = pd.Series(den).rolling(window=4, min_periods=4).mean().values
    rvi = rvi_num / rvi_den
    
    # Signal line: 4-period SMA of RVI
    rvi_signal = pd.Series(rvi).rolling(window=4, min_periods=4).mean().values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(rvi[i]) or np.isnan(rvi_signal[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg_20[i]
        
        # RVI signals: RVI crosses above/below signal line
        rvi_cross_up = rvi[i-1] <= rvi_signal[i-1] and rvi[i] > rvi_signal[i]
        rvi_cross_down = rvi[i-1] >= rvi_signal[i-1] and rvi[i] < rvi_signal[i]
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: RVI crosses up AND bullish trend AND volume confirmation
        if rvi_cross_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: RVI crosses down AND bearish trend AND volume confirmation
        elif rvi_cross_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite RVI cross
        elif position == 1 and rvi_cross_down:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rvi_cross_up:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals