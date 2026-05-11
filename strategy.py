#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for entry with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend. Exits on opposite Camarilla level touch.
Designed to work in both bull and bear markets by following 12h trend and using volatility-adaptive entries.
Targets ~25-35 trades/year via tight Camarilla breakout conditions + trend + volume filters.
"""

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    # Pivot Point
    pivot = (high + low + close) / 3
    # Range
    range_ = high - low
    
    # Resistance levels
    r1 = close + (range_ * 1.1 / 12)
    r2 = close + (range_ * 1.1 / 6)
    r3 = close + (range_ * 1.1 / 4)
    r4 = close + (range_ * 1.1 / 2)
    
    # Support levels
    s1 = close - (range_ * 1.1 / 12)
    s2 = close - (range_ * 1.1 / 6)
    s3 = close - (range_ * 1.1 / 4)
    s4 = close - (range_ * 1.1 / 2)
    
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla Pivot Levels ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Handle first day
    prev_high[0] = df_1d['high'].iloc[0]
    prev_low[0] = df_1d['low'].iloc[0]
    prev_close[0] = df_1d['close'].iloc[0]
    
    _, r1, _, _, _, s1, _, _, _ = calculate_camarilla(prev_high, prev_low, prev_close)
    
    # Align daily Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # --- 12h EMA50 Trend Filter ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema_50_12h_4h = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h trend: 1 if price above EMA50, -1 if below
    trend_12h = np.where(close_12h > ema_50_12h, 1, -1)
    trend_12h_4h = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # --- Volume Spike Detection (24-period average = 2 days on 4h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.divide(volume, vol_ma, out=np.ones_like(volume), where=vol_ma!=0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_50_12h_4h[i]) or np.isnan(trend_12h_4h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # 12h trend direction
        trend = trend_12h_4h[i]
        
        if position == 0:
            # Long: 12h uptrend + price breaks above R1 + volume
            if (trend == 1 and 
                close[i] > r1_4h[i] and 
                close[i-1] <= r1_4h[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: 12h downtrend + price breaks below S1 + volume
            elif (trend == -1 and 
                  close[i] < s1_4h[i] and 
                  close[i-1] >= s1_4h[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price touches S1 (opposite level)
                if close[i] <= s1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price touches R1 (opposite level)
                if close[i] >= r1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals