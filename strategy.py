#!/usr/bin/env python3

name = "1h_Camarilla_R1S1_Breakout_4h1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA30 for trend filter
    ema_30_4h = pd.Series(df_4h['close'].values).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_30_4h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous day
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_R1 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_S1 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: current volume > 2.0x 24-period average (1h)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (2.0 * vol_ma_24)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Breakout threshold
    breakout_threshold = 0.003  # 0.3%
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # 6 hours cooldown to reduce trades
    
    start_idx = max(30, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema_30_4h_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 4h trend direction
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h['close'].values)
        trend_4h_up = close_4h_aligned[i] > ema_30_4h_aligned[i]
        trend_4h_down = close_4h_aligned[i] < ema_30_4h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Check session filter
            if not session_filter[i]:
                continue
                
            # Long: Break above R1 with volume in 4h uptrend
            if (close[i] > R1_aligned[i] * (1 + breakout_threshold) and 
                close[i-1] <= R1_aligned[i-1] and 
                trend_4h_up and 
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S1 with volume in 4h downtrend
            elif (close[i] < S1_aligned[i] * (1 - breakout_threshold) and 
                  close[i-1] >= S1_aligned[i-1] and 
                  trend_4h_down and 
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Close below S1 OR trend change
            if (close[i] < S1_aligned[i]) or not trend_4h_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Close above R1 OR trend change
            if (close[i] > R1_aligned[i]) or not trend_4h_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation.
# Long when price breaks above R1 level (with 0.3% buffer) with volume spike in 4h uptrend during active session (08-20 UTC).
# Short when price breaks below S1 level (with 0.3% buffer) with volume spike in 4h downtrend during active session.
# Uses 1d Camarilla levels for intraday support/resistance and 4h EMA30 for trend.
# Volume confirmation and session filter reduce false signals. 6-hour cooldown limits trades to 15-35/year.
# Designed to work in both bull and bear markets by following higher timeframe trend.