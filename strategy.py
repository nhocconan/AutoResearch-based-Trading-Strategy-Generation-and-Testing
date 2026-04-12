#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (OHLC from previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC (Camarilla uses previous day's range)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for previous day
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    # H2 = Close + 0.75 * (High - Low)
    # L2 = Close - 0.75 * (High - Low)
    # H1 = Close + 0.5 * (High - Low)
    # L1 = Close - 0.5 * (High - Low)
    
    range_1d = prev_high - prev_low
    h4 = prev_close + 1.5 * range_1d
    l4 = prev_close - 1.5 * range_1d
    h3 = prev_close + 1.125 * range_1d
    l3 = prev_close - 1.125 * range_1d
    h2 = prev_close + 0.75 * range_1d
    l2 = prev_close - 0.75 * range_1d
    h1 = prev_close + 0.5 * range_1d
    l1 = prev_close - 0.5 * range_1d
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, l2)
    h1_aligned = align_htf_to_ltf(prices, df_1d, h1)
    l1_aligned = align_htf_to_ltf(prices, df_1d, l1)
    
    # Volume filter: 24-period average on 12h data (2 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50-period EMA on 1d close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(h1_aligned[i]) or np.isnan(l1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals with volume confirmation
        # Long: Break above H3 or H4 with volume
        long_signal = ((close[i] > h3_aligned[i] or close[i] > h4_aligned[i]) and 
                      uptrend and volume_ok[i])
        # Short: Break below L3 or L4 with volume
        short_signal = ((close[i] < l3_aligned[i] or close[i] < l4_aligned[i]) and 
                       downtrend and volume_ok[i])
        
        # Exit when price returns to H1/L1 levels (mean reversion to mean)
        exit_long = close[i] < h1_aligned[i]
        exit_short = close[i] > l1_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals