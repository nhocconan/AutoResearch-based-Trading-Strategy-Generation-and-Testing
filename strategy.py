#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 level AND 1d close > 1d EMA34 AND volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S3 level AND 1d close < 1d EMA34 AND volume > 2.0x 20-period MA.
Exit when price reverses to Camarilla pivot point (PP).
Designed for low trade frequency (target: 20-50/year) with trend following in 4h timeframe.
Camarilla levels provide precise intraday support/resistance, daily trend filter ensures alignment with higher timeframe momentum.
Volume spike confirmation reduces false breakouts. Strategy should work in both bull and bear markets by
trading breakouts in the direction of the daily trend.
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
    
    # Calculate Camarilla levels for 4h timeframe
    # Based on previous period's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = pivot + (range_hl * 1.1 / 4)
    S3 = pivot - (range_hl * 1.1 / 4)
    PP = pivot  # Pivot point for exit
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (spike confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 level AND uptrend AND volume spike
            if close[i] > R3[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 level AND downtrend AND volume spike
            elif close[i] < S3[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price reverses to Camarilla pivot point (PP)
            exit_signal = False
            
            if position == 1:
                # Long exit: price crosses below pivot point
                if close[i] < PP[i]:
                    exit_signal = True
            elif position == -1:
                # Short exit: price crosses above pivot point
                if close[i] > PP[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0