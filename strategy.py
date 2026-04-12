#!/usr/bin/env python3
"""
1h_4d_camarilla_breakout_volume_filter
Hypothesis: 1-hour Camarilla breakout with 4-hour volume confirmation and daily volatility filter.
Uses higher timeframes for signal direction (4h trend, 1d volatility regime) and 1h for precise entry timing.
Designed to work in both bull and bear markets by avoiding false breakouts via volume and volatility filters.
Target: 15-37 trades/year (60-150 total over 4 years) to minimize fee drag.
"""

name = "1h_4d_camarilla_breakout_volume_filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend direction (using close price for EMA)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h EMA21 for trend direction
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get daily data for volatility filter (ATR) and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Previous day's range for Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first bar
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high_1d[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low_1d[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close_1d[0]
    
    # Camarilla levels (based on previous day)
    range_ = prev_high - prev_low
    # Resistance levels
    r3 = prev_close + range_ * 1.1 / 2
    r4 = prev_close + range_ * 1.1
    # Support levels
    s3 = prev_close - range_ * 1.1 / 2
    s4 = prev_close - range_ * 1.1
    
    # Daily ATR for volatility filter (14-day ATR)
    tr1 = np.abs(np.subtract(high_1d, low_1d))
    tr2 = np.abs(np.subtract(high_1d, np.roll(close_1d, 1)))
    tr3 = np.abs(np.subtract(low_1d, np.roll(close_1d, 1)))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # ATR as percentage of price for normalization
    atr_pct = atr_1d / np.maximum(close_1d, 1e-8)
    
    # Daily volume average (20-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d data to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    atr_pct_aligned = align_htf_to_ltf(prices, df_1d, atr_pct)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = max(50, 21)  # EMA21 needs 21 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(atr_pct_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long entry: price above 4h EMA21 (uptrend), breaks above R4 with volume expansion
        if (close[i] > ema_4h_aligned[i] and  # 4h uptrend filter
            close[i] > r4_aligned[i] and     # Break above R4
            volume[i] > vol_ma_1d_aligned[i] * 1.5 and  # Volume > 1.5x daily average
            atr_pct_aligned[i] > 0.01 and    # Minimum volatility filter (1% ATR)
            position != 1):
            position = 1
            signals[i] = 0.20
        
        # Short entry: price below 4h EMA21 (downtrend), breaks below S4 with volume expansion
        elif (close[i] < ema_4h_aligned[i] and  # 4h downtrend filter
              close[i] < s4_aligned[i] and      # Break below S4
              volume[i] > vol_ma_1d_aligned[i] * 1.5 and  # Volume > 1.5x daily average
              atr_pct_aligned[i] > 0.01 and     # Minimum volatility filter (1% ATR)
              position != -1):
            position = -1
            signals[i] = -0.20
        
        # Exit conditions
        elif position == 1 and close[i] < s3_aligned[i]:  # Close back below S3
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > r3_aligned[i]:  # Close back above R3
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals