#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla R3/S3 breakout on 1d timeframe with 1w EMA50 trend filter and volume confirmation (>1.8x 20-period MA).
Long when price breaks above R3 with uptrend and volume spike.
Short when price breaks below S3 with downtrend and volume filter.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to capture strong breakouts in both bull and bear markets by aligning with weekly trend.
Target: 7-25 trades/year (30-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Get 1d data for Camarilla calculation (use previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 1.8x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1w EMA + 20 for volume MA)
    start_idx = 70
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Need previous day's OHLC for Camarilla (1d data lagged by 1 bar)
        if i >= 1:
            idx_1d = i // 24  # Approximate 1d bar index from 1h bars, but we use HTF data properly
            # Actually, we need to access the previous completed 1d bar
            # Since we're on 1d timeframe, we can use direct indexing with lag
            if i >= 1:
                prev_high = df_1d['high'].values[i-1] if i-1 < len(df_1d) else df_1d['high'].values[-1]
                prev_low = df_1d['low'].values[i-1] if i-1 < len(df_1d) else df_1d['low'].values[-1]
                prev_close = df_1d['close'].values[i-1] if i-1 < len(df_1d) else df_1d['close'].values[-1]
                
                # Calculate Camarilla levels for current day based on previous day
                range_val = prev_high - prev_low
                if range_val > 0:
                    R3 = prev_close + range_val * 1.1 / 4
                    S3 = prev_close - range_val * 1.1 / 4
                    
                    if position == 0:
                        # Long: price breaks above R3 with 1w uptrend and volume spike
                        if close[i] > R3 and uptrend_1w[i] and volume_spike[i]:
                            signals[i] = 0.25
                            position = 1
                        # Short: price breaks below S3 with 1w downtrend and volume spike
                        elif close[i] < S3 and downtrend_1w[i] and volume_spike[i]:
                            signals[i] = -0.25
                            position = -1
                        else:
                            signals[i] = 0.0
                    elif position == 1:
                        # Hold long
                        signals[i] = 0.25
                        # Exit: price closes below R3 or trend changes to downtrend
                        if close[i] < R3 or not uptrend_1w[i]:
                            signals[i] = 0.0
                            position = 0
                    elif position == -1:
                        # Hold short
                        signals[i] = -0.25
                        # Exit: price closes above S3 or trend changes to uptrend
                        if close[i] > S3 or not downtrend_1w[i]:
                            signals[i] = 0.0
                            position = 0
                else:
                    signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0