#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA50 trend filter and volume confirmation
- Uses Camarilla pivot levels from 4h timeframe for stronger breakout signals (R4/S4 vs R3/S3)
- 12h EMA50 defines higher timeframe trend: only trade breakouts in trend direction
- Volume confirmation (> 1.8x 20-period average) filters false breakouts
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading with the 12h trend
- Camarilla R4/S4 levels provide stronger breakout signals than R3/S3 with fewer whipsaws
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
    
    # Calculate 4h Camarilla pivot levels (R4, S4)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: R4 = close + (high - low) * 1.1/2, S4 = close - (high - low) * 1.1/2
    camarilla_r4_4h = close_4h + (high_4h - low_4h) * 1.1 / 2
    camarilla_s4_4h = close_4h - (high_4h - low_4h) * 1.1 / 2
    
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h)
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: > 1.8x 20-period average (balanced to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 with 12h uptrend and volume spike
            long_breakout = (close[i] > camarilla_r4_aligned[i] and 
                           close[i] > ema_50_12h_aligned[i] and
                           volume[i] > 1.8 * vol_ma[i])
            
            # Short conditions: price breaks below Camarilla S4 with 12h downtrend and volume spike
            short_breakout = (close[i] < camarilla_s4_aligned[i] and 
                            close[i] < ema_50_12h_aligned[i] and
                            volume[i] > 1.8 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Camarilla S4 or 12h trend turns bearish
                if (close[i] < camarilla_s4_aligned[i] or 
                    close[i] < ema_50_12h_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Camarilla R4 or 12h trend turns bullish
                if (close[i] > camarilla_r4_aligned[i] or 
                    close[i] > ema_50_12h_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R4S4_Breakout_12hEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0