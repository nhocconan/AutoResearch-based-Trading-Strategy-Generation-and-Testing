# 6h Camarilla R4/S4 Breakout with Weekly Trend and Volume Spike
# Hypothesis: 6h timeframe with 12h/1d HTF filters. Uses weekly pivot for trend direction
# and 12h for entry timing with volume confirmation. Designed for low trade frequency
# (target 20-50/year) to avoid fee drag while capturing strong breakouts in both bull/bear markets.
# Weekly trend provides macro bias, 12h R4/S4 breakouts with volume capture momentum,
# and mean reversion at R3/S3 prevents whipsaws in ranging markets.

#!/usr/bin/env python3
name = "6h_Camarilla_R4S4_WeeklyTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for entry levels (R4/S4)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1d data for weekly trend (using daily closes to build weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h R4/S4 levels for breakout
    prev_high_12h = df_12h['high'].values
    prev_low_12h = df_12h['low'].values
    prev_close_12h = df_12h['close'].values
    range_12h = prev_high_12h - prev_low_12h
    r4 = prev_close_12h + range_12h * 1.1 / 2 * 1.1  # R4 = C + (H-L) * 1.1/2 * 1.1
    s4 = prev_close_12h - range_12h * 1.1 / 2 * 1.1  # S4 = C - (H-L) * 1.1/2 * 1.1
    
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4)
    
    # Calculate weekly trend from daily data: weekly EMA(8) on daily closes
    # Weekly trend: if price > weekly EMA, bullish; else bearish
    weekly_ema = pd.Series(df_1d['close'].values).ewm(span=8, adjust=False, min_periods=8).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1d, weekly_ema)
    
    # Volume filter: spike detection (2.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: avoid low volatility (ATR > 0.4% of price)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    vol_filter = atr > 0.004 * close  # ATR > 0.4% of price
    
    # Session filter: 08:00 - 20:00 UTC (avoid low-volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure we have volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(weekly_ema_aligned[i]) or np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(vol_filter[i]) or not vol_filter[i] or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation
        volume_filter = volume[i] > 2.5 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R4 + weekly uptrend + volume spike
            if (close[i] > r4_12h_aligned[i] and 
                close[i] > weekly_ema_aligned[i] and   # Weekly uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S4 + weekly downtrend + volume spike
            elif (close[i] < s4_12h_aligned[i] and 
                  close[i] < weekly_ema_aligned[i] and   # Weekly downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to weekly EMA (mean reversion)
            if (position == 1 and close[i] < weekly_ema_aligned[i]) or \
               (position == -1 and close[i] > weekly_ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals