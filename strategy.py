#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivot levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (high, low, close)
    # Use previous day's data to avoid look-ahead
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Previous day's OHLC
        prev_high = df_1d['high'].values[i-1] if i-1 < len(df_1d) else np.nan
        prev_low = df_1d['low'].values[i-1] if i-1 < len(df_1d) else np.nan
        prev_close = df_1d['close'].values[i-1] if i-1 < len(df_1d) else np.nan
        
        if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
            range_val = prev_high - prev_low
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2
    
    # Volume filter: current volume > 1.5x 24-period average (for 4h, 24*4h = 4d)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~1 day for 4h to reduce trades
    
    start_idx = max(100, 24, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_s4[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 1d trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R3 with volume in uptrend
            if (close[i] > camarilla_r3[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume in downtrend
            elif (close[i] < camarilla_s3[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below Camarilla S3 or trend changes
            if close[i] < camarilla_s3[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above Camarilla R3 or trend changes
            if close[i] > camarilla_r3[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation on 4h timeframe.
# Long when price breaks above Camarilla R3 level in uptrend with volume confirmation.
# Short when price breaks below Camarilla S3 level in downtrend with volume confirmation.
# Uses 4h timeframe to balance trade frequency and capture meaningful trends.
# Target: 75-200 total trades over 4 years (19-50/year) as per experiment guidelines.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Based on top-performing pattern from DB: Camarilla breakout + volume + trend filter.