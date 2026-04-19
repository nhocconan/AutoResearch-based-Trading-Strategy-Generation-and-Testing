#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h ATR breakout with 1d trend filter and volume confirmation.
# Uses ATR-based breakout from 20-period high/low to capture momentum.
# 1d EMA50 filters trend direction (long only above EMA50, short only below).
# Volume > 1.5x 20-period average confirms breakout strength.
# Designed for fewer trades (target: 20-40/year) with clear trend/filter logic.
# Works in bull/bear markets by requiring both trend alignment and volatility expansion.
name = "12h_ATRBreakout_EMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ATR calculation (14-period) for breakout bands
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], 
                       abs(high[i] - close[i-1]), 
                       abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]  # first TR
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align 1d EMA50 to 12h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure EMA50 and 20-period channels are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_50_val = ema_50_aligned[i]
        high_break = high_20[i]
        low_break = low_20[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long: price breaks above 20-period high, above EMA50, volume confirmed
            if price > high_break and price > ema_50_val and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low, below EMA50, volume confirmed
            elif price < low_break and price < ema_50_val and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 20-period low or closes below EMA50
            if price < low_break or price < ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-period high or closes above EMA50
            if price > high_break or price > ema_50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals