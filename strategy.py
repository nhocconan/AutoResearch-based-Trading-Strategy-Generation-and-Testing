#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Camarilla R1/S1 breakout on 12h timeframe with 1d trend filter and volume confirmation.
Long when price breaks above R1 in uptrend (1d close > 1d EMA34), short when breaks below S1 in downtrend.
Exits when price re-enters Camarilla H3/L3 range or trend reverses.
Designed for low trade frequency (12-37/year) via strict breakout conditions and trend filter.
Works in both bull and bear markets by following 1d trend direction.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close
    Returns: (R4, R3, R2, R1, PP, S1, S2, S3, S4)
    """
    typical = (high + low + close) / 3
    range_ = high - low
    
    R1 = close + range_ * 1.1 / 12
    R2 = close + range_ * 1.1 / 6
    R3 = close + range_ * 1.1 / 4
    R4 = close + range_ * 1.1 / 2
    
    PP = (high + low + close) / 3
    
    S1 = close - range_ * 1.1 / 12
    S2 = close - range_ * 1.1 / 6
    S3 = close - range_ * 1.1 / 4
    S4 = close - range_ * 1.1 / 2
    
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Trend Filter ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_1d = np.where(close_1d > ema_34_1d, 1, -1)
    
    # Align 1d trend to 12h
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # --- 12h Camarilla Levels ---
    # Need previous day's HLC for today's Camarilla
    # For 12h timeframe, we use prior 12h bar's HLC
    R4, R3, R2, R1, PP, S1, S2, S3, S4 = calculate_camarilla(high, low, close)
    
    # Shift to get prior bar's levels (avoid look-ahead)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R3_prev = np.roll(R3, 1)
    S3_prev = np.roll(S3, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    R3_prev[0] = np.nan
    S3_prev[0] = np.nan
    
    # Volume Spike Detection (24-period average = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(R1_prev[i]) or 
            np.isnan(S1_prev[i]) or np.isnan(R3_prev[i]) or np.isnan(S3_prev[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.8
        
        # 1d trend direction
        daily_trend = trend_1d_aligned[i]
        
        if position == 0:
            # Long: daily uptrend + price breaks above R1 + volume
            if (daily_trend == 1 and 
                close[i] > R1_prev[i] and 
                close[i-1] <= R1_prev[i-1] and  # crossed above this bar
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: daily downtrend + price breaks below S1 + volume
            elif (daily_trend == -1 and 
                  close[i] < S1_prev[i] and 
                  close[i-1] >= S1_prev[i-1] and  # crossed below this bar
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: daily trend turns down OR price re-enters S3
                if (trend_1d_aligned[i] == -1 or 
                    close[i] < S3_prev[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: daily trend turns up OR price re-enters R3
                if (trend_1d_aligned[i] == 1 or 
                    close[i] > R3_prev[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals