#!/usr/bin/env python3
"""
1h_4d_Camarilla_Breakout_TrendFilter_Volume
Hypothesis: On 1h timeframe, use 4h Camarilla levels (R3/S3) for breakout entries and (R1/S1) for exits,
filtered by 4h EMA50 trend and volume confirmation. Trades only during 08-20 UTC to avoid low-volume hours.
Targets 15-30 trades/year per symbol (60-120 over 4 years) to minimize fee drag.
Works in bull/bear by following 4h trend direction and requiring volume confirmation.
"""

name = "1h_4d_Camarilla_Breakout_TrendFilter_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 4h data for trend, pivot levels, and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    volume_1h = prices['volume'].values
    
    # --- 4h Trend Filter: EMA50 ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- 4h Volume Confirmation: 20-period average ---
    volume_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    # --- Camarilla Pivots from 4h (previous bar) ---
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivots from previous 4h bar's data
    camarilla_high = np.full_like(close_4h, np.nan)
    camarilla_low = np.full_like(close_4h, np.nan)
    camarilla_close = np.full_like(close_4h, np.nan)
    
    for i in range(1, len(close_4h)):
        camarilla_high[i] = high_4h[i-1]
        camarilla_low[i] = low_4h[i-1]
        camarilla_close[i] = close_4h[i-1]
    
    # Calculate Camarilla levels
    R4 = camarilla_close + ((camarilla_high - camarilla_low) * 1.5000)
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    R2 = camarilla_close + ((camarilla_high - camarilla_low) * 1.1666)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    PP = camarilla_close
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    S2 = camarilla_close - ((camarilla_high - camarilla_low) * 1.1666)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    S4 = camarilla_close - ((camarilla_high - camarilla_low) * 1.5000)
    
    # Align pivots to 1h timeframe
    R3_1h = align_htf_to_ltf(prices, df_4h, R3)
    S3_1h = align_htf_to_ltf(prices, df_4h, S3)
    R1_1h = align_htf_to_ltf(prices, df_4h, R1)
    S1_1h = align_htf_to_ltf(prices, df_4h, S1)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical values are NaN
        if (np.isnan(R3_1h[i]) or np.isnan(S3_1h[i]) or 
            np.isnan(R1_1h[i]) or np.isnan(S1_1h[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close_1h[i] > ema50_4h_aligned[i]
        trend_down = close_1h[i] < ema50_4h_aligned[i]
        
        # Volume confirmation (using 4h volume)
        vol_ok = volume_1h[i] > vol_ma_20_4h_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume
            if close_1h[i] > R3_1h[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 4h uptrend + volume
                signals[i] = 0.20
                position = 1
            elif close_1h[i] < S3_1h[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 4h downtrend + volume
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close_1h[i] <= S1_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close_1h[i] >= R1_1h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals