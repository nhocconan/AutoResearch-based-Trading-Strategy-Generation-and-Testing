#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 1-day data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    camarilla_high = np.full(n, np.nan)
    camarilla_low = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous day's data (already complete)
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        
        if pd.isna(prev_high) or pd.isna(prev_low) or pd.isna(prev_close):
            continue
            
        range_val = prev_high - prev_low
        camarilla_high[i] = prev_close + range_val * 1.1 / 2
        camarilla_low[i] = prev_close - range_val * 1.1 / 2
        r3[i] = prev_close + range_val * 1.1 * 3 / 4
        s3[i] = prev_close - range_val * 1.1 * 3 / 4
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~6 hours for 4h
    
    start_idx = max(100, 20, 50, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(r3[i]) or 
            np.isnan(s3[i]) or 
            np.isnan(camarilla_high[i]) or 
            np.isnan(camarilla_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine 12h trend direction
        trend_up = close > ema_50_12h_aligned[i]
        trend_down = close < ema_50_12h_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above R3 with volume in uptrend
            if (close[i] > r3[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below S3 with volume in downtrend
            elif (close[i] < s3[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below camarilla low or trend changes
            if close[i] < camarilla_low[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above camarilla high or trend changes
            if close[i] > camarilla_high[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation on 4h timeframe.
# Long when price breaks above R3 in uptrend with volume confirmation.
# Short when price breaks below S3 in downtrend with volume confirmation.
# Uses Camarilla levels from daily pivot points for institutional reference points.
# Includes cooldown period to reduce trade frequency and avoid overtrading.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Target: 75-200 total trades over 4 years (19-50/year) as per experiment guidelines.