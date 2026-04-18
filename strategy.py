#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for 40-period high/low (channel) and 14-period ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 40-period highest high and lowest low on daily
    highest_40 = pd.Series(high_1d).rolling(window=40, min_periods=40).max().values
    lowest_40 = pd.Series(low_1d).rolling(window=40, min_periods=40).min().values
    
    # Calculate 14-period ATR on daily
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align daily channel and ATR to 4h timeframe
    highest_40_aligned = align_htf_to_ltf(prices, df_1d, highest_40)
    lowest_40_aligned = align_htf_to_ltf(prices, df_1d, lowest_40)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h ATR for position sizing
    tr_4h_1 = high - low
    tr_4h_2 = np.abs(high - np.roll(close, 1))
    tr_4h_3 = np.abs(low - np.roll(close, 1))
    tr_4h_1[0] = high[0] - low[0]
    tr_4h_2[0] = np.abs(high[0] - close[0])
    tr_4h_3[0] = np.abs(low[0] - close[0])
    tr_4h = np.maximum(tr_4h_1, np.maximum(tr_4h_2, tr_4h_3))
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # need daily channel, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_40_aligned[i]) or np.isnan(lowest_40_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_4h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: price breaks above 40-day high with volume
            if close[i] > highest_40_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 40-day low with volume
            elif close[i] < lowest_40_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price closes below 40-day low or ATR trailing stop
            if close[i] < lowest_40_aligned[i] or close[i] < high[i] - 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above 40-day high or ATR trailing stop
            if close[i] > highest_40_aligned[i] or close[i] > low[i] + 2.0 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DailyChannelBreakout_Volume_ATRTrailing"
timeframe = "4h"
leverage = 1.0