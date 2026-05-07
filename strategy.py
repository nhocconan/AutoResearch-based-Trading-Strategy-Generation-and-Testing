#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1/2
    # S3 = C - (H - L) * 1.1/2
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r3 = daily_close + (daily_high - daily_low) * 1.1 / 2.0
    s3 = daily_close - (daily_high - daily_low) * 1.1 / 2.0
    
    # Align daily Camarilla levels to 12h timeframe (available after daily bar closes)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 1  # ~12 hours to prevent overtrading
    
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above daily R3 with volume in uptrend
            if (close[i] > r3_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.30
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below daily S3 with volume in downtrend
            elif (close[i] < s3_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.30
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls back below daily pivot or trend changes
            if close[i] < pivot_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: Price rises back above daily pivot or trend changes
            if close[i] > pivot_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.30
    
    return signals

# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: On 12h timeframe, price breaking above/below daily Camarilla R3/S3 levels with volume confirmation and daily trend filter captures institutional order flow. Camarilla R3/S3 act as key support/resistance where breakouts often continue in the direction of the daily trend. This approach works in bull markets (breakouts above R3 in uptrend) and bear markets (breakdowns below S3 in downtrend) by aligning with higher timeframe momentum. Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag while capturing significant moves.