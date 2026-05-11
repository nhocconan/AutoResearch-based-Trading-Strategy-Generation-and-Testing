#!/usr/bin/env python3
"""
1h_4d_1d_Camarilla_Pivot_Breakout_Volume_Trend
Hypothesis: Use 1d and 4h timeframes for signal direction, 1h only for precise entry timing.
- Long when: price breaks above R3 (1d) with volume > 20-period average and 4h close > 4h EMA50
- Short when: price breaks below S3 (1d) with volume > 20-period average and 4h close < 4h EMA50
- Exit when price returns to opposite pivot level (S1 for longs, R1 for shorts)
- Session filter: only trade 08-20 UTC to avoid low-liquidity periods
- Position size: 0.20 (20% of capital) to manage drawdown
Target: 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
"""

name = "1h_4d_1d_Camarilla_Pivot_Breakout_Volume_Trend"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Pivot Levels (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots from previous day's OHLC
    camarilla_high = np.full_like(close_1d, np.nan)
    camarilla_low = np.full_like(close_1d, np.nan)
    camarilla_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        camarilla_high[i] = high_1d[i-1]
        camarilla_low[i] = low_1d[i-1]
        camarilla_close[i] = close_1d[i-1]
    
    # Calculate Camarilla levels
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    
    # Align pivots to 1h timeframe
    R3_1h = align_htf_to_ltf(prices, df_1d, R3)
    S3_1h = align_htf_to_ltf(prices, df_1d, S3)
    R1_1h = align_htf_to_ltf(prices, df_1d, R1)
    S1_1h = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- 4h Trend Filter: EMA50 ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- Volume Confirmation: 1h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # --- Session Filter: 08-20 UTC ---
    hours = prices.index.hour  # pre-compute before loop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 100  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_1h[i]) or np.isnan(S3_1h[i]) or 
            np.isnan(R1_1h[i]) or np.isnan(S1_1h[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close[i] > ema50_4h_aligned[i]
        trend_down = close[i] < ema50_4h_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume
            if close[i] > R3_1h[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 4h uptrend + volume
                signals[i] = 0.20
                position = 1
            elif close[i] < S3_1h[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 4h downtrend + volume
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close[i] <= S1_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close[i] >= R1_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals