#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla levels on daily data for better stability
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: H4/L4 (resistance/support)
    # H4 = Close + 1.1/2 * (High - Low)
    # L4 = Close - 1.1/2 * (High - Low)
    camarilla_h4 = prev_close + 1.1/2 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1/2 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume filter - 24-period average (1 day on 1h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > vol_ma
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after volume MA warmup
        # Skip if not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Skip outside session
        if not session_ok[i]:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Breakout signals with volume confirmation
        # Long: price breaks above H4 resistance
        long_signal = close[i] > camarilla_h4_aligned[i] and volume_ok[i]
        # Short: price breaks below L4 support
        short_signal = close[i] < camarilla_l4_aligned[i] and volume_ok[i]
        
        # Exit when price returns to midpoint (mean reversion)
        midpoint = (camarilla_h4_aligned[i] + camarilla_l4_aligned[i]) / 2
        exit_long = close[i] < midpoint and position == 1
        exit_short = close[i] > midpoint and position == -1
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals