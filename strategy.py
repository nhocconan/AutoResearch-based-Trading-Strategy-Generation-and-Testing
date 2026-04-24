#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h for precise entry timing
- HTF: 4h for EMA50 trend direction and 1d for Camarilla pivot levels
- Entry: Long when price breaks above R3 with 4h uptrend and volume spike (>2x 20-period average)
         Short when price breaks below S3 with 4h downtrend and volume spike
- Exit: When price returns to Camarilla pivot (PP) or volume drops below average
- Signal size: 0.20 discrete levels to minimize fee churn
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag
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
    
    # Get 1d data for Camarilla pivot calculation (standard lookback period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous 1d bar
    # Using typical Camarilla formula based on previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3/S3 are the strongest breakout levels
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # PP = (high + low + close) / 3 (pivot point)
    range_1d = high_1d - low_1d
    r3 = close_1d + 1.1 * range_1d / 2
    s3 = close_1d - 1.1 * range_1d / 2
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Align HTF levels to LTF (1h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Get 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Trend filter: price above/below 4h EMA50
    uptrend = close > ema_50_4h_aligned
    downtrend = close < ema_50_4h_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (avoid low liquidity hours)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 AND uptrend AND volume spike
            if close[i] > r3_aligned[i] and uptrend[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND downtrend AND volume spike
            elif close[i] < s3_aligned[i] and downtrend[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot (PP) OR no volume spike
            if close[i] <= pp_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to pivot (PP) OR no volume spike
            if close[i] >= pp_aligned[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0