#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h trend filter and 1d volume confirmation. 
In bull/bear markets: breakouts aligned with 4h trend + above-average 1d volume capture strong moves while avoiding false breakouts in chop. 
Uses discrete sizing (0.20) and session filter (08-20 UTC) to minimize fee churn. Targets 60-150 trades over 4 years by requiring confluence of 3 factors.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    # Load 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR for Camarilla levels (using 5-period ATR)
    atr_period = 5
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Previous day's OHLC for Camarilla calculation (using 4h data as proxy for daily)
    # We'll use the last 4h bar's OHLC to approximate previous day
    prev_close = np.roll(close, 1)  # previous close
    prev_high = np.roll(high, 1)    # previous high
    prev_low = np.roll(low, 1)      # previous low
    
    # Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Using previous day's range
    camarilla_range = prev_high - prev_low
    R1 = prev_close + 1.1 * camarilla_range / 12
    S1 = prev_close - 1.1 * camarilla_range / 12
    
    # Calculate 4h EMA20 for trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    htf_trend = np.where(close > ema_20_4h_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    # Calculate 1d average volume for volume filter
    avg_vol_1d = pd.Series(df_1d['volume'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    volume_filter = volume > avg_vol_1d_aligned  # above average volume
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for EMA, 1 for Camarilla)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Check session
        if not session_filter[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Long entry: price breaks above R1 with 4h uptrend and volume confirmation
        if close[i] > R1[i] and htf_trend[i] == 1 and volume_filter[i]:
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        # Short entry: price breaks below S1 with 4h downtrend and volume confirmation
        elif close[i] < S1[i] and htf_trend[i] == -1 and volume_filter[i]:
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        # Exit conditions: price returns to previous close (mean reversion)
        elif position == 1 and close[i] < prev_close[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > prev_close[i]:
            signals[i] = 0.0
            position = 0
        # Hold current position
        else:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter_v1"
timeframe = "1h"
leverage = 1.0