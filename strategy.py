#!/usr/bin/env python3
name = "12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 12h Camarilla levels using previous 12h high/low/close
    # Camarilla formula: 
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12)
    # S1 = C - ((H-L) * 1.1/12)
    # S2 = C - ((H-L) * 1.1/6)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # We focus on R3 and S3 for breakouts
    
    # Shift high/low/close by 1 to use previous bar's values
    high_prev = np.roll(high, 1)
    low_prev = np.roll(low, 1)
    close_prev = np.roll(close, 1)
    high_prev[0] = high[0]  # first bar placeholder
    low_prev[0] = low[0]
    close_prev[0] = close[0]
    
    H = high_prev
    L = low_prev
    C = close_prev
    
    R3 = C + ((H - L) * 1.1 / 4)
    S3 = C - ((H - L) * 1.1 / 4)
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: > 1.5x 24-period average (12 days for 12h timeframe)
    vol_ma_12h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 1.5 * vol_ma_12h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_12h[i]) or
            np.isnan(R3[i]) or np.isnan(S3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with uptrend and volume
            if (close[i] > R3[i] and close[i] > ema_34_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with downtrend and volume
            elif (close[i] < S3[i] and close[i] < ema_34_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below EMA34 or breaks below S3 (reversal)
            if close[i] < ema_34_1d_aligned[i] or close[i] < S3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above EMA34 or breaks above R3 (reversal)
            if close[i] > ema_34_1d_aligned[i] or close[i] > R3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 levels provide strong support/resistance.
# Breakouts above R3 (bullish) or below S3 (bearish) with 1d EMA34 trend filter
# and volume confirmation capture institutional flow. Works in both bull and bear
# markets by following the 1d trend while using 12h structure for precise entries.
# Target: 20-30 trades/year to minimize fee drag. Position size 0.25 limits risk.