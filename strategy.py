#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 4h close > EMA50 AND volume > 1.5x 20-period average.
Short when price breaks below Camarilla S1 AND 4h close < EMA50 AND volume > 1.5x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for longs, R1 for shorts).
Uses 4h HTF for EMA50 trend direction (avoids whipsaws in counter-trend moves). Target: 60-150 total trades over 4 years (15-37/year).
Camarilla levels provide precise intraday support/resistance; 4h EMA50 ensures we trade with the intermediate trend.
Works in bull markets (long bias above EMA50) and bear markets (short bias below EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels (based on previous bar's range)
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Use previous bar to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # first bar uses current close
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_range = prev_high - prev_low
    camarilla_R1 = prev_close + 1.1 * camarilla_range / 12
    camarilla_S1 = prev_close - 1.1 * camarilla_range / 12
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA (20), 4h EMA calculation (50)
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i] or (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_R1[i]) or 
                                 np.isnan(camarilla_S1[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r1 = camarilla_R1[i]
        s1 = camarilla_S1[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above Camarilla R1 AND 4h close > EMA50 AND volume spike
            if price > r1 and price > ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.20
                position = 1
            # Short: Break below Camarilla S1 AND 4h close < EMA50 AND volume spike
            elif price < s1 and price < ema_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.20
                position = -1
        else:
            # Exit when price touches opposite Camarilla level
            if position == 1 and price < s1:  # Long exit at Camarilla S1
                exit_signal = True
            elif position == -1 and price > r1:  # Short exit at Camarilla R1
                exit_signal = True
            else:
                exit_signal = False
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R1S1_Breakout_4hEMA50_Trend_VolumeConfirmation_LevelExit"
timeframe = "1h"
leverage = 1.0