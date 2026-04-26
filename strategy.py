#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h EMA50 trend filter and volume spike confirmation.
Uses 1h timeframe for entry timing, 4h for trend direction and Camarilla levels, 1d for volume context.
Targets 15-37 trades/year (60-150 total over 4 years) by requiring confluence of pivot break, HTF trend, and volume spike.
Position size 0.20 to manage drawdown in bear markets. Includes session filter (08-20 UTC) to avoid low-liquidity hours.
Designed to work in both bull and bear markets via trend filter and strict entry conditions.
"""

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
    
    # Get 4h data for HTF trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA(50) for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Get 1d volume median for volume context (optional filter)
    df_1d_vol = get_htf_data(prices, '1d')
    if len(df_1d_vol) < 2:
        vol_median_1d = np.full(n, np.nan)
    else:
        vol_median_1d = pd.Series(df_1d_vol['volume'].values).rolling(window=20, min_periods=20).median().values
    
    # Align HTF indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    vol_median_1d_aligned = align_htf_to_ltf(prices, df_1d_vol, vol_median_1d) if len(df_1d_vol) >= 2 else np.full(n, np.nan)
    
    # 1h volume confirmation: 2.0x rolling median
    vol_median_1h = pd.Series(volume).rolling(window=24, min_periods=24).median().values  # 24 * 1h = 1 day
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 4h EMA (50), 1h volume median (24), 1d volume median (20)
    start_idx = max(50, 24, 20)
    
    # Precompute session hours for 08-20 UTC filter
    hours = prices.index.hour
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_median_1h[i]) or 
            np.isnan(vol_median_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_1h_val = vol_median_1h[i]
        vol_median_1d_val = vol_median_1d_aligned[i]
        
        if position == 0:
            # Long: break above R1, uptrend (close > EMA50), volume spike vs 1h and 1d median
            long_signal = (high_val > camarilla_r1_val) and \
                          (close_val > ema_50_4h_val) and \
                          (volume_val > 2.0 * vol_median_1h_val) and \
                          (volume_val > 1.5 * vol_median_1d_val)  # Additional 1d volume context
            # Short: break below S1, downtrend (close < EMA50), volume spike vs 1h and 1d median
            short_signal = (low_val < camarilla_s1_val) and \
                           (close_val < ema_50_4h_val) and \
                           (volume_val > 2.0 * vol_median_1h_val) and \
                           (volume_val > 1.5 * vol_median_1d_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period to reduce churn
            bars_since_entry += 1
            signals[i] = 0.20
            # Exit: trend reversal (close < EMA50) after minimum holding period
            if bars_since_entry >= 6 and (close_val < ema_50_4h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.20
            # Exit: trend reversal (close > EMA50) after minimum holding period
            if bars_since_entry >= 6 and (close_val > ema_50_4h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0