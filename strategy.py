#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot R1/S1 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND 4h close > EMA34 AND volume > 1.5x 20-bar average.
Short when price breaks below S1 AND 4h close < EMA34 AND volume > 1.5x 20-bar average.
Exit when price touches the opposite pivot level (S1 for longs, R1 for shorts).
Uses 4h HTF for trend alignment (avoids counter-trend whipsaws). Target: 80-150 total trades over 4 years (20-37/year).
Camarilla pivots provide precise intraday support/resistance; EMA34 filter ensures we trade with the 4h trend.
Works in both bull and bear markets by only taking trend-aligned breaks.
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (precompute hour array)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA34 for trend filter (HTF) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h Camarilla pivots (based on previous bar's OHLC)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close + 1.1 * (high - low) / 12
    camarilla_s1 = close - 1.1 * (high - low) / 12
    
    # Shift pivots by 1 to avoid look-ahead (use previous bar's pivots)
    camarilla_r1 = np.roll(camarilla_r1, 1)
    camarilla_s1 = np.roll(camarilla_s1, 1)
    camarilla_r1[0] = np.nan
    camarilla_s1[0] = np.nan
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # volume MA (20), EMA calculation (34)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_4h_aligned[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above R1 AND 4h close > EMA34 AND volume spike
            if price > r1 and close[i] > ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below S1 AND 4h close < EMA34 AND volume spike
            elif price < s1 and close[i] < ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit when price touches opposite pivot level
            if position == 1 and price < s1:  # Long exit at S1
                exit_signal = True
            elif position == -1 and price > r1:  # Short exit at R1
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeConfirmation_LevelExit"
timeframe = "1h"
leverage = 1.0