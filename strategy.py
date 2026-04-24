#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla Pivot Breakout with 4h Trend Filter and Volume Spike.
- Uses 4h EMA50 for trend direction (price > EMA50 = uptrend, price < EMA50 = downtrend)
- 1h Camarilla levels: Long on break above R3, Short on break below S3
- Volume confirmation: current volume > 2.0 * 20-period average volume
- Session filter: 08-20 UTC only to avoid low-liquidity Asian session
- Designed to work in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend)
- Camarilla pivots provide structured support/resistance; EMA50 filters trend; volume confirms conviction
- Signal size: 0.20 discrete levels to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h trend: price > EMA50 = uptrend, price < EMA50 = downtrend
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # Calculate 1h Camarilla pivots (using previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    camarilla_span = 1.1 * (high - low)
    r3 = close + camarilla_span
    s3 = close - camarilla_span
    
    # Breakout conditions: price breaks above R3 (long) or below S3 (short)
    breakout_long = high > r3
    breakout_short = low < s3
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = open_time.dt.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # volume MA and 4h EMA need warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not session_filter.iloc[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 AND uptrend on 4h AND volume confirmation
            if breakout_long[i] and uptrend_4h[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 AND downtrend on 4h AND volume confirmation
            elif breakout_short[i] and downtrend_4h[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 (reversal) OR loss of uptrend
            if low[i] < s3[i] or not uptrend_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R3 (reversal) OR loss of downtrend
            if high[i] > r3[i] or not downtrend_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0