#!/usr/bin/env python3
"""
6h_1d_Camarilla_R4S4_Breakout_Volume_EMA34Filter_v1
Hypothesis: Breakout at extreme Camarilla levels (R4/S4) on 6h with 1d EMA34 trend filter and volume confirmation.
Works in bull/bear: In uptrend, buy R4 breakout; in downtrend, sell S4 breakdown. Uses 1d EMA34 for trend, volume for confirmation.
Target: 12-25 trades/year per symbol (50-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R4, S4 (extreme breakout levels)
    rang = prev_high - prev_low
    r4 = prev_close + rang * 6.0 / 12
    s4 = prev_close - rang * 6.0 / 12
    
    # Align to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: price > R4 (breakout) AND 1d uptrend AND volume
            if (price > r4_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and  # 1d EMA rising
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < S4 (breakdown) AND 1d downtrend AND volume
            elif (price < s4_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and  # 1d EMA falling
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 1d EMA34 (trend reversal) or price < R3 (mean reversion)
            # Calculate R3 for exit
            prev_high_i = high_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_low_i = low_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_close_i = close_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            if not (np.isnan(prev_high_i) or np.isnan(prev_low_i) or np.isnan(prev_close_i)):
                rang_i = prev_high_i - prev_low_i
                r3_exit = prev_close_i + rang_i * 3.0 / 12
                # Align R3 exit level (simplified: use current day's R3)
                r3_exit_aligned = align_htf_to_ltf(prices, df_1d, 
                                                  pd.Series([prev_close_i + rang_i * 3.0 / 12] * len(df_1d)).values)
                if price < ema_34_1d_aligned[i] or (not np.isnan(r3_exit_aligned[i]) and price < r3_exit_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 1d EMA34 (trend reversal) or price > S3 (mean reversion)
            prev_high_i = high_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_low_i = low_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            prev_close_i = close_1d[np.searchsorted(df_1d.index, prices['open_time'].iloc[i]) - 1] if i > 0 else np.nan
            if not (np.isnan(prev_high_i) or np.isnan(prev_low_i) or np.isnan(prev_close_i)):
                rang_i = prev_high_i - prev_low_i
                s3_exit = prev_close_i - rang_i * 3.0 / 12
                # Align S3 exit level (simplified: use current day's S3)
                s3_exit_aligned = align_htf_to_ltf(prices, df_1d, 
                                                  pd.Series([prev_close_i - rang_i * 3.0 / 12] * len(df_1d)).values)
                if price > ema_34_1d_aligned[i] or (not np.isnan(s3_exit_aligned[i]) and price > s3_exit_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R4S4_Breakout_Volume_EMA34Filter_v1"
timeframe = "6h"
leverage = 1.0