#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (H3/L3) breakout with daily EMA34 trend filter and volume confirmation.
# Camarilla levels provide high-probability reversal/breakout zones based on prior day's range.
# Daily EMA34 ensures we trade in the direction of the higher timeframe trend.
# Volume confirmation adds conviction to breakouts.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above H3) and bear markets (breakouts below L3).
name = "4h_Camarilla_H3L3_DailyEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and EMA filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (H3, L3) from previous day's data
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    range_prev = high_prev - low_prev
    H3 = close_prev + range_prev * 1.1 / 6
    L3 = close_prev - range_prev * 1.1 / 6
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(close_prev).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 4h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above H3 AND EMA34 uptrend AND volume confirmation
            long_breakout = close[i] > H3_aligned[i]
            uptrend = close[i] > ema_34_aligned[i]
            if vol_confirm and long_breakout and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND EMA34 downtrend AND volume confirmation
            elif vol_confirm and close[i] < L3_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR EMA34 turns down
            exit_condition = close[i] < L3_aligned[i] or close[i] < ema_34_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR EMA34 turns up
            exit_condition = close[i] > H3_aligned[i] or close[i] > ema_34_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals