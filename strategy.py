#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_trend_v1
# Strategy: 1d Camarilla pivot levels with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H3/L3, H4/L4) act as strong support/resistance.
# Long when price crosses above H3 with weekly uptrend (price > weekly EMA200) and volume confirmation.
# Short when price crosses below L3 with weekly downtrend (price < weekly EMA200) and volume confirmation.
# Weekly trend filter reduces whipsaws in sideways markets. Works in bull via continuation at support,
# in bear via resistance bounces. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Daily high, low, close for Camarilla calculation
    high_1d = df_1w['high'].values  # Use weekly high/low for Camarilla? No - need daily
    # Actually need daily OHLC for Camarilla - get 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    # Camarilla formulas:
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    prev_high = df_1d['high'].values
    prev_low = df_1d['low'].values
    prev_close = df_1d['close'].values
    
    # Shift to get previous day's values
    prev_high = np.roll(prev_high, 1)
    prev_low = np.roll(prev_low, 1)
    prev_close = np.roll(prev_close, 1)
    prev_high[0] = prev_high[1]  # fill first value
    prev_low[0] = prev_low[1]
    prev_close[0] = prev_close[1]
    
    high_low_range = prev_high - prev_low
    H3 = prev_close + 1.1 * high_low_range
    L3 = prev_close - 1.1 * high_low_range
    H4 = prev_close + 1.5 * high_low_range
    L4 = prev_close - 1.5 * high_low_range
    
    # Align Camarilla levels to daily timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_avg_20_aligned[i]
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Entry conditions
        # Long: price crosses above H3, weekly uptrend, volume confirmation
        if (close[i] > H3_aligned[i] and close[i-1] <= H3_aligned[i-1] and 
            weekly_uptrend and vol_confirm and position != 1):
            position = 1
            signals[i] = 0.25
        # Short: price crosses below L3, weekly downtrend, volume confirmation
        elif (close[i] < L3_aligned[i] and close[i-1] >= L3_aligned[i-1] and 
              weekly_downtrend and vol_confirm and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price crosses opposite level (H4 for long, L4 for short) or trend change
        elif position == 1 and (close[i] < L3_aligned[i] or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > H3_aligned[i] or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals