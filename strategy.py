#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level breakout with 12h EMA trend filter and volume confirmation
# Camarilla pivot levels (R1/S1) identify key support/resistance levels where price often reverses or breaks.
# Breakout above R1 or below S1 with volume confirmation captures momentum moves in both bull and bear markets.
# 12h EMA filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# Designed for 4h timeframe targeting 20-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    R1 = close_1d + (range_hl * 1.1 / 12)
    S1 = close_1d - (range_hl * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + 12h uptrend + volume confirmation
            if (close[i] > R1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + 12h downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price returns below S1 or trend turns down
                if (close[i] < S1_aligned[i] or 
                    close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns above R1 or trend turns up
                if (close[i] > R1_aligned[i] or 
                    close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0