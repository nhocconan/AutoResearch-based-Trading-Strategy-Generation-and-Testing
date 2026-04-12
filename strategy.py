#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_v31"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # H4 = C + 1.5 * (H - L)
    # L4 = C - 1.5 * (H - L)
    # H3 = C + 1.25 * (H - L)
    # L3 = C - 1.25 * (H - L)
    # H2 = C + 1.0833 * (H - L)
    # L2 = C - 1.0833 * (H - L)
    # H1 = C + 1.0833 * (H - L) / 2
    # L1 = C - 1.0833 * (H - L) / 2
    # P = (H + L + C) / 3
    
    # Calculate for previous day (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value invalid
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    H4 = prev_close + 1.5 * (prev_high - prev_low)
    L4 = prev_close - 1.5 * (prev_high - prev_low)
    H3 = prev_close + 1.25 * (prev_high - prev_low)
    L3 = prev_close - 1.25 * (prev_high - prev_low)
    H2 = prev_close + 1.0833 * (prev_high - prev_low)
    L2 = prev_close - 1.0833 * (prev_high - prev_low)
    H1 = prev_close + 1.0833 * (prev_high - prev_low) / 2
    L1 = prev_close - 1.0833 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Volume filter: 20-period average on 4h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50-period EMA on 1d close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(H2_aligned[i]) or np.isnan(L2_aligned[i]) or
            np.isnan(H1_aligned[i]) or np.isnan(L1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions
        # Long: Break above H3 with volume confirmation in uptrend
        long_breakout = close[i] > H3_aligned[i] and volume_ok[i] and uptrend
        # Short: Break below L3 with volume confirmation in downtrend
        short_breakout = close[i] < L3_aligned[i] and volume_ok[i] and downtrend
        
        # Exit conditions
        # Exit long: Price returns below H1 or trend changes
        exit_long = close[i] < H1_aligned[i] or not uptrend
        # Exit short: Price returns above L1 or trend changes
        exit_short = close[i] > L1_aligned[i] or not downtrend
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
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