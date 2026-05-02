#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation
# Uses 6h timeframe to balance trade frequency and signal quality
# Elder Ray measures bull/bear power relative to EMA13 to detect institutional buying/selling pressure
# 12h EMA50 provides intermediate trend filter to avoid counter-trend trades
# Volume confirmation ensures breakouts have participation
# Works in both bull and bear markets by following 12h trend while using Elder Ray for timing
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate 1w HTF trend (using close price EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d HTF trend (using close price EMA34) for additional filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine overall trend: both 1w and 1d EMA34 must agree with 12h EMA50
        bullish_trend = (ema_34_1w_aligned[i] > ema_34_1d_aligned[i] and 
                        close[i] > ema_50_12h_aligned[i])
        bearish_trend = (ema_34_1w_aligned[i] < ema_34_1d_aligned[i] and 
                        close[i] < ema_50_12h_aligned[i])
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power positive (buying pressure) with volume spike AND bullish trend
            if (bull_power[i] > 0 and 
                volume_spike[i] and 
                bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short entry: Bear Power negative (selling pressure) with volume spike AND bearish trend
            elif (bear_power[i] < 0 and 
                  volume_spike[i] and 
                  bearish_trend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative OR bearish trend develops
            if bull_power[i] <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR bullish trend develops
            if bear_power[i] >= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals