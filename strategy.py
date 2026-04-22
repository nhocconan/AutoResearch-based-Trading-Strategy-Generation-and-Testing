#!/usr/bin/env python3
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
    
    # Load 12h data for trend and 1d for support/resistance
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 1d Support/Resistance: Previous day's high/low
    support_1d = low_1d
    resistance_1d = high_1d
    support_aligned = align_htf_to_ltf(prices, df_1d, support_1d)
    resistance_aligned = align_htf_to_ltf(prices, df_1d, resistance_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(support_aligned[i]) or 
            np.isnan(resistance_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above 1d resistance + above 12h EMA20 + volume spike
            if close[i] > resistance_aligned[i] and close[i] > ema_20_12h_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below 1d support + below 12h EMA20 + volume spike
            elif close[i] < support_aligned[i] and close[i] < ema_20_12h_aligned[i] and volume[i] > 1.5 * vol_avg_20[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 12h EMA20 in opposite direction
            if position == 1:
                # Exit long: Close below 12h EMA20
                if close[i] < ema_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Close above 12h EMA20
                if close[i] > ema_20_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_12hEMA20_1dSupportResistance_Breakout_Volume"
timeframe = "4h"
leverage = 1.0