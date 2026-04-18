#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (H4/L4) breakout with 12h trend filter and volume confirmation.
# Camarilla levels identify key support/resistance based on prior day's range.
# 12h EMA34 filters trades to align with medium-term trend.
# Volume confirmation ensures breakout conviction.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above H4 in uptrend) and bear markets (breakouts below L4 in downtrend).
name = "4h_Camarilla_H4L4_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (prior day's range)
    df_1d = get_htf_data(prices, '1d')
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla levels (H4, L4) using prior day's OHLC
    # H4 = C + (H-L) * 1.1/2
    # L4 = C - (H-L) * 1.1/2
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_h4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_l4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (available after daily close)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_period = 34
    ema_12h = pd.Series(close_12h).ewm(span=ema_period, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
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
            # Long: price breaks above H4 AND 12h EMA34 is rising (uptrend) AND volume confirmation
            long_breakout = close[i] > camarilla_h4_aligned[i]
            ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1] if i > 0 else False
            if vol_confirm and ema_rising and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 AND 12h EMA34 is falling (downtrend) AND volume confirmation
            elif vol_confirm and not ema_rising and close[i] < camarilla_l4_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L4 OR 12h EMA34 turns down
            exit_condition = close[i] < camarilla_l4_aligned[i] or ema_12h_aligned[i] < ema_12h_aligned[i-1]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H4 OR 12h EMA34 turns up
            exit_condition = close[i] > camarilla_h4_aligned[i] or ema_12h_aligned[i] > ema_12h_aligned[i-1]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals