#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol
# Hypothesis: Camarilla pivot breakout on 1h with 4h trend filter and 1d volume confirmation.
# Uses 4h close vs 4h EMA50 for trend direction, 1h Camarilla R1/S1 breakout for entry,
# and 1d volume > 20-day average for conviction. Designed for 15-30 trades/year to avoid
# fee drag while capturing breakouts in both bull and bear markets via trend alignment.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA50 for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d, additional_delay_bars=0)
    
    # Calculate 1h Camarilla levels (R1, S1) from previous day
    # Using daily high, low, close from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_width = (high_1d - low_1d) * 1.1 / 12
    r1_1d = close_1d + camarilla_width
    s1_1d = close_1d - camarilla_width
    
    # Align Camarilla levels to 1h (no delay needed as they are based on prior day)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup for indicators
        # Skip if any critical value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or vol_ma_20_1d_aligned[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above 4h EMA50 (uptrend), and volume confirmation
            if (close[i] > r1_1d_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and
                volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, below 4h EMA50 (downtrend), and volume confirmation
            elif (close[i] < s1_1d_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  volume[i] > vol_ma_20_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 or trend changes (below 4h EMA50)
            if (close[i] < s1_1d_aligned[i] or 
                close[i] < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price breaks above R1 or trend changes (above 4h EMA50)
            if (close[i] > r1_1d_aligned[i] or 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals