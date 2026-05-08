#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PriceAction_SwingRejection_1dTrend_Volume"
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
    
    # 1d trend: EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d swing points for dynamic support/resistance
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # Swing high: higher high than previous and next bar
    swing_high = np.zeros_like(high_1d)
    swing_low = np.zeros_like(low_1d)
    for i in range(1, len(high_1d)-1):
        if high_1d[i] > high_1d[i-1] and high_1d[i] > high_1d[i+1]:
            swing_high[i] = high_1d[i]
        if low_1d[i] < low_1d[i-1] and low_1d[i] < low_1d[i+1]:
            swing_low[i] = low_1d[i]
    # Forward fill to get last swing level
    swing_high = pd.Series(swing_high).replace(0, np.nan).ffill().bfill().values
    swing_low = pd.Series(swing_low).replace(0, np.nan).ffill().bfill().values
    
    swing_high_aligned = align_htf_to_ltf(prices, df_1d, swing_high)
    swing_low_aligned = align_htf_to_ltf(prices, df_1d, swing_low)
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(swing_high_aligned[i]) or 
            np.isnan(swing_low_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: rejection of swing low support + uptrend + volume
            long_cond = (low[i] <= swing_low_aligned[i] and close[i] > swing_low_aligned[i]) and \
                        (close[i] > ema_50_1d_aligned[i]) and \
                        volume_filter[i]
            # Short: rejection of swing high resistance + downtrend + volume
            short_cond = (high[i] >= swing_high_aligned[i] and close[i] < swing_high_aligned[i]) and \
                         (close[i] < ema_50_1d_aligned[i]) and \
                         volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below swing low or trend reversal
            if close[i] < swing_low_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above swing high or trend reversal
            if close[i] > swing_high_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals