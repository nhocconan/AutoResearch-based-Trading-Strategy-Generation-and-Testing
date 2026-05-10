#!/usr/bin/env python3
# 1h_4h1d_Trend_Filter_with_Camarilla_Entry
# Hypothesis: Combining 4h trend (EMA34) and 1d momentum (ROC5) filters with Camarilla breakouts on 1h
# provides institutional-level entries with trend alignment, working in both bull and bear markets.
# Uses volume confirmation (1.5x average) to filter false breakouts.
# Target: 15-35 trades/year to minimize fee drag on 1h timeframe.

name = "1h_4h1d_Trend_Filter_with_Camarilla_Entry"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_4h_up = close_4h > ema34_4h
    trend_4h_down = close_4h < ema34_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1d momentum filter (ROC5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 6:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    roc5_1d = np.zeros_like(close_1d)
    roc5_1d[5:] = (close_1d[5:] - close_1d[:-5]) / close_1d[:-5] * 100
    mom_1d_up = roc5_1d > 0
    mom_1d_down = roc5_1d < 0
    
    # Align 1d momentum to 1h
    mom_1d_up_aligned = align_htf_to_ltf(prices, df_1d, mom_1d_up.astype(float))
    mom_1d_down_aligned = align_htf_to_ltf(prices, df_1d, mom_1d_down.astype(float))
    
    # Volume confirmation (1.5x 24-period average)
    vol_ma = np.zeros_like(volume)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma[i] = vol_sum / 24
        else:
            vol_ma[i] = np.nan
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Calculate Camarilla levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift to get previous 4h bar values
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels (R3, S3, R4, S4)
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    R4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    R4_aligned = align_htf_to_ltf(prices, df_4h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_4h, S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(mom_1d_up_aligned[i]) or np.isnan(mom_1d_down_aligned[i]) or
            np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation, 4h uptrend, and 1d momentum up
            if (high[i] > R3_aligned[i] and
                trend_4h_up_aligned[i] > 0.5 and
                mom_1d_up_aligned[i] > 0.5 and
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume confirmation, 4h downtrend, and 1d momentum down
            elif (low[i] < S3_aligned[i] and
                  trend_4h_down_aligned[i] > 0.5 and
                  mom_1d_down_aligned[i] > 0.5 and
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: price breaks below S3 or 4h trend turns down
            if (low[i] < S3_aligned[i] or
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: price breaks above R3 or 4h trend turns up
            if (high[i] > R3_aligned[i] or
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals