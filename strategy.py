#!/usr/bin/env python3
"""
1d Keltner Channel Breakout with 1w EMA Trend Filter and Volume Confirmation
Captures volatility expansion phases in both bull and bear markets.
Buy when price breaks above upper Keltner channel in uptrend with volume.
Sell when price breaks below lower Keltner channel in downtrend with volume.
Uses ATR-based channels that adapt to volatility, reducing false signals.
Target: 20-25 trades/year to minimize fee drag while capturing meaningful moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Keltner Channel (20, 2.0) on 1d
    kc_length = 20
    kc_mult = 2.0
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full(n, np.nan)
    for i in range(kc_length, n):
        atr[i] = np.nanmean(tr[i-kc_length+1:i+1])
    
    # Calculate EMA of close for Keltner middle
    ema_close = np.full(n, np.nan)
    if n >= kc_length:
        ema_close[kc_length-1] = np.mean(close[:kc_length])
        multiplier = 2 / (kc_length + 1)
        for i in range(kc_length, n):
            ema_close[i] = (close[i] * multiplier) + (ema_close[i-1] * (1 - multiplier))
    
    # Keltner Bands
    kc_middle = ema_close
    kc_upper = kc_middle + kc_mult * atr
    kc_lower = kc_middle - kc_mult * atr
    
    # Get 1w data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close
    ema_50_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_50_1w[49] = np.mean(close_1w[:50])
        multiplier_1w = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_50_1w[i] = (close_1w[i] * multiplier_1w) + (ema_50_1w[i-1] * (1 - multiplier_1w))
    
    # Align 1w EMA to 1d timeframe
    ema_50_1w_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_length, 20)  # need KC, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_1w_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1w EMA50
        trend_up = close[i] > ema_50_1w_1d[i]
        trend_down = close[i] < ema_50_1w_1d[i]
        
        if position == 0:
            # Long entry: close above upper KC with volume and uptrend
            if (close[i] > kc_upper[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below lower KC with volume and downtrend
            elif (close[i] < kc_lower[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below middle KC or reverse signal
            if close[i] < kc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above middle KC or reverse signal
            if close[i] > kc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KeltnerChannelBreakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0