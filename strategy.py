#!/usr/bin/env python3
# 1h_4h_1d_camarilla_pivot_volume_v1
# Strategy: 1-hour Camarilla pivot breakout with 4-hour trend filter and daily volume confirmation
# Timeframe: 1h
# Leverage: 1.0
# Hypothesis: Camarilla pivot levels (H3/L3) act as strong intraday support/resistance.
# Long when price breaks above H3 with 4h uptrend (price > EMA50) and volume confirmation (VOL > 1.5x 20-period average).
# Short when price breaks below L3 with 4h downtrend (price < EMA50) and volume confirmation.
# Works in bull markets by capturing breakouts and in bear markets by catching breakdowns with volume.
# Uses 4h for trend direction, 1h for entry timing, and daily volume filter to reduce false signals.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_pivot_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (active trading hours)
    hours = prices.index.hour
    
    # Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Calculate Camarilla levels for 1h (using previous bar's OHLC)
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    # We use previous bar's data to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = close[0]  # Initialize first value
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):  # Start from 1 to have previous bar data
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0  # Force flat outside session
            continue
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_h3[i]
        breakout_down = close[i] < camarilla_l3[i]
        
        # Trend filter: 4h EMA50
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume filter: current 1h volume > 1.5x daily 20-period average volume
        vol_confirm = volume[i] > (1.5 * vol_avg_20_1d_aligned[i])
        
        # Entry logic: breakout + trend + volume + session
        if breakout_up and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.20
        elif breakout_down and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.20
        # Exit: opposite breakout with volume confirmation
        elif position == 1 and breakout_down and vol_confirm:
            position = 0
            signals[i] = 0.0
        elif position == -1 and breakout_up and vol_confirm:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals