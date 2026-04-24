#!/usr/bin/env python3
"""
Hypothesis: 1h 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h for entry timing, using 4h for signal direction (Camarilla levels)
- HTF: 1d EMA50 for trend filter (bullish if close > EMA50, bearish if close < EMA50)
- Long when price breaks above 4h Camarilla H3 AND 1d EMA50 trend up AND volume > 1.5 * median volume of last 20 bars
- Short when price breaks below 4h Camarilla L3 AND 1d EMA50 trend down AND volume > 1.5 * median volume of last 20 bars
- Exit on opposite Camarilla breakout or trend reversal (close crosses 1d EMA50)
- Uses discrete position size 0.20 to minimize fee churn
- Session filter: 08-20 UTC to avoid low-volume Asian session noise
- Target: 60-150 total trades over 4 years (15-37/year) to stay within fee drag limits
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
    
    # Calculate 4h Camarilla levels (based on previous 4h bar's range)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get previous 4h bar's OHLC
    prev_4h_close = df_4h['close'].shift(1).values
    prev_4h_high = df_4h['high'].shift(1).values
    prev_4h_low = df_4h['low'].shift(1).values
    
    # Calculate Camarilla H3/L3 for 4h
    camarilla_h3_4h = prev_4h_close + 1.1 * (prev_4h_high - prev_4h_low) / 4
    camarilla_l3_4h = prev_4h_close - 1.1 * (prev_4h_high - prev_4h_low) / 4
    
    # Align 4h Camarilla levels to 1h timeframe
    camarilla_h3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 1.5 * median volume of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (1.5 * vol_median)
    
    # Session filter: 08-20 UTC (avoid low-volume Asian session)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h3_4h_aligned[i]) or np.isnan(camarilla_l3_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_median[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla H3, 1d EMA50 trend up, volume confirmation
            if close[i] > camarilla_h3_4h_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla L3, 1d EMA50 trend down, volume confirmation
            elif close[i] < camarilla_l3_4h_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 4h Camarilla L3 OR trend reversal (close < 1d EMA50)
            if close[i] < camarilla_l3_4h_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 4h Camarilla H3 OR trend reversal (close > 1d EMA50)
            if close[i] > camarilla_h3_4h_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hDir_1dEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0