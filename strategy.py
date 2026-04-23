#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 1d EMA34 uptrend AND volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S3 AND 1d EMA34 downtrend AND volume > 2.0x 20-period MA.
Exit when price crosses Camarilla pivot point (PP) or opposite Camarilla level break.
Designed for ~20-40 trades/year with strong trend-following edge in both bull and bear markets.
Camarilla levels provide high-probability reversal/breakout points based on prior day's range.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (using 1d data aligned to 4h)
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # We need prior 1d bar's H,L,C to calculate levels for current 4h period
    df_1d_prev = df_1d.copy()
    # Shift 1d data by 1 bar to get prior completed day
    high_1d_prev = np.roll(df_1d_prev['high'].values, 1)
    low_1d_prev = np.roll(df_1d_prev['low'].values, 1)
    close_1d_prev = np.roll(df_1d_prev['close'].values, 1)
    # First bar will have invalid data (rolled from last), but min_periods will handle via alignment
    
    camarilla_pp = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    camarilla_r3 = close_1d_prev + (high_1d_prev - low_1d_prev) * 1.1 / 2.0
    camarilla_s3 = close_1d_prev - (high_1d_prev - low_1d_prev) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34, volume MA20 (Camarilla uses 1d data which is aligned)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_r3 = close[i] > camarilla_r3_aligned[i]  # break above R3
        breakout_s3 = close[i] < camarilla_s3_aligned[i]  # break below S3
        pp_cross = (position == 1 and close[i] < camarilla_pp_aligned[i]) or \
                   (position == -1 and close[i] > camarilla_pp_aligned[i])
        opposite_breakout = (position == 1 and breakout_s3) or \
                            (position == -1 and breakout_r3)
        
        if position == 0:
            # Long: Camarilla R3 breakout AND uptrend AND volume confirmation
            if breakout_r3 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Camarilla S3 breakout AND downtrend AND volume confirmation
            elif breakout_s3 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: pivot point cross or opposite Camarilla level break
            exit_signal = pp_cross or opposite_breakout
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0