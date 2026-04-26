#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation_v4
Hypothesis: Further tighten entry conditions from v3 to reduce trade frequency and avoid overtrading.
- Increase volume confirmation threshold from 1.5x to 2.0x 20-period MA
- Require both volume spike AND price closing above/below Camarilla level for confirmation
- Add ATR-based volatility filter: only trade when ATR(14) > 0.5 * ATR(50) to avoid low-volatility chop
- Keep discrete position sizing at 0.25 to minimize fee churn
- Target: 15-30 trades/year (60-120 total over 4 years) to stay well under 400 max
- Works in both bull and bear markets by following 1d EMA34 trend
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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's OHLC for current day's levels
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Camarilla R1, S1, R3, S3 levels
    camarilla_range = high_1d_prev - low_1d_prev
    r1 = close_1d_prev + camarilla_range * 1.1 / 12
    s1 = close_1d_prev - camarilla_range * 1.1 / 12
    r3 = close_1d_prev + camarilla_range * 1.1 / 4
    s3 = close_1d_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA (tightened from 1.5x to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR-based volatility filter: avoid low-volatility chop
    atr_14 = pd.Series(close).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(np.diff(x, prepend=x[0]))), raw=False
    ).values
    atr_50 = pd.Series(close).rolling(window=50, min_periods=50).apply(
        lambda x: np.mean(np.abs(np.diff(x, prepend=x[0]))), raw=False
    ).values
    volatility_filter = atr_14 > (0.5 * atr_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for ATR + 34 for 1d EMA + 20 for volume MA + 1 for Camarilla shift)
    start_idx = 105
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(volatility_filter[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price closes above R1 with 1d uptrend, volume spike, and volatility filter
            if (close[i] > r1_aligned[i] and 
                uptrend_1d[i] and volume_spike[i] and volatility_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price closes below S1 with 1d downtrend, volume spike, and volatility filter
            elif (close[i] < s1_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i] and volatility_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R3 (strong reversal) OR 1d trend changes to downtrend
            if (close[i] < r3_aligned[i] or not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S3 (strong reversal) OR 1d trend changes to uptrend
            if (close[i] > s3_aligned[i] or not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation_v4"
timeframe = "4h"
leverage = 1.0