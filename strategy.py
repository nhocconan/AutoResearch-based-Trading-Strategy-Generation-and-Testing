#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND 1d EMA34 is rising AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S1 AND 1d EMA34 is falling AND volume > 1.8x 20-period average.
Exit when price touches the opposite Camarilla level (S1 for long, R1 for short) or EMA34 reverses.
Uses 1d HTF for EMA34 trend to avoid whipsaws in ranging markets. Target: 75-200 total trades over 4 years.
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
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (based on previous day's OHLC)
    # We need daily OHLC to compute Camarilla levels for intraday
    # Since we're on 4h timeframe, we approximate using rolling window
    # But proper Camarilla requires daily OHLC - we'll use 6-period (1.5 days) as proxy
    lookback = 6  # Approximately 1.5 days of 4h data
    pivots = np.full(n, np.nan)
    r1 = np.full(n, np.nan)
    s1 = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        # Use highest high, lowest low, and close of the lookback period as proxy for daily OHLC
        period_high = np.max(high[i - lookback + 1:i + 1])
        period_low = np.min(low[i - lookback + 1:i + 1])
        period_close = close[i]
        
        # Camarilla pivot point calculation
        pivots[i] = (period_high + period_low + period_close) / 3
        range_val = period_high - period_low
        r1[i] = pivots[i] + range_val * 1.1 / 12
        s1[i] = pivots[i] - range_val * 1.1 / 12
    
    # 20-period volume average for spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback - 1, 34, 20)  # Camarilla (6), EMA34 (34), volume MA (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(pivots[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_val = ema_34_aligned[i]
        r1_val = r1[i]
        s1_val = s1[i]
        vol_ma_val = vol_ma[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: Break above Camarilla R1 AND EMA34 rising AND volume spike
            if price > r1_val and ema_rising and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S1 AND EMA34 falling AND volume spike
            elif price < s1_val and ema_falling and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S1 OR EMA34 starts falling
                if price < s1_val or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R1 OR EMA34 starts rising
                if price > r1_val or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0