#!/usr/bin/env python3
# 12h_Camarilla_Pivot_1w_Trend_Volume_v2
# Hypothesis: Camarilla pivot levels from weekly data with 1d EMA trend filter and volume confirmation.
# Uses 12h timeframe to reduce trade frequency. Weekly pivots provide strong support/resistance.
# Trend filter ensures alignment with daily momentum. Volume confirms breakout strength.
# Designed for 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "12h_Camarilla_Pivot_1w_Trend_Volume_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla pivot levels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    # Calculate Camarilla levels from previous weekly bar
    # H, L, C from previous weekly candle
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Camarilla levels: 
    # Resistance: R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*0.5/2
    # Support: S1 = C - (H-L)*0.5/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    # But we only need S3, S2, S1, R1, R2, R3 for trading
    # Actually standard Camarilla uses: 
    # R4 = C + (H-L)*1.5/2, R3 = C + (H-L)*1.25/2, R2 = C + (H-L)*1.1/2, R1 = C + (H-L)*0.5/2
    # S1 = C - (H-L)*0.5/2, S2 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.25/2, S4 = C - (H-L)*1.5/2
    
    # We'll use R3, R2, S2, S3 as key levels
    camarilla_r3 = weekly_close + (weekly_high - weekly_low) * 1.25 / 2
    camarilla_r2 = weekly_close + (weekly_high - weekly_low) * 1.1 / 2
    camarilla_s2 = weekly_close - (weekly_high - weekly_low) * 1.1 / 2
    camarilla_s3 = weekly_close - (weekly_high - weekly_low) * 1.25 / 2
    
    # Align weekly Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r3)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s2)
    s3_aligned = align_htf_to_ltf(prices, df_weekly, camarilla_s3)
    
    # Daily EMA trend filter (34-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 24-period average (~12 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    if n >= vol_period:
        vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(24, 1)  # volume period
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S2 or trend fails
            if close[i] < s2_aligned[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R2 or trend fails
            if close[i] > r2_aligned[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above R2 with uptrend
                if close[i] > r2_aligned[i] and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below S2 with downtrend
                elif close[i] < s2_aligned[i] and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals