#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot Range Breakout with 1-day Trend Filter and Volume Spike.
Long when price breaks above R3 with 1-day EMA34 uptrend and volume spike.
Short when price breaks below S3 with 1-day EMA34 downtrend and volume spike.
Exit when price returns to the central pivot point (P).
Camarilla levels provide intraday support/resistance; 1-day EMA ensures higher-timeframe trend alignment;
volume spike confirms institutional interest. Designed for low trade frequency by requiring breakouts
beyond extreme levels (R3/S3) rather than inner ranges. Works in both bull and bear markets by following
the 1-day trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using daily data to get prior day's range
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day's OHLC (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    P = (prev_high + prev_low + prev_close) / 3
    R3 = P + (range_ * 1.1 / 2)
    S3 = P - (range_ * 1.1 / 2)
    R4 = P + (range_ * 1.1)
    S4 = P - (range_ * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # 1-day trend filter: EMA34 on close
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(P_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above R3 with 1-day EMA34 uptrend and volume spike
            if (close[i] > R3_aligned[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with 1-day EMA34 downtrend and volume spike
            elif (close[i] < S3_aligned[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to central pivot point (P)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below P
                if close[i] < P_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above P
                if close[i] > P_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0