#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
Long when price breaks above R3 in 1d uptrend with volume > 2.0x 20-period MA.
Short when price breaks below S3 in 1d downtrend with volume > 2.0x 20-period MA.
Exit when price touches R4/S4 or returns to 1d EMA34.
Uses 1d HTF for trend alignment to reduce whipsaw. Designed for ~15-30 trades/year with strong edge via institutional pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close, close, close
    close_val = close
    R4 = close_val + range_val * 1.1 / 2
    R3 = close_val + range_val * 1.1 / 4
    R2 = close_val + range_val * 1.1 / 6
    R1 = close_val + range_val * 1.1 / 12
    PP = (high + low + close_val) / 3
    S1 = close_val - range_val * 1.1 / 12
    S2 = close_val - range_val * 1.1 / 6
    S3 = close_val - range_val * 1.1 / 4
    S4 = close_val - range_val * 1.1 / 2
    return R4, R3, R2, R1, PP, S1, S2, S3, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Calculate Camarilla levels from 1d OHLC
    camarilla_data = []
    for i in range(len(df_1d)):
        h = df_1d['high'].iloc[i]
        l = df_1d['low'].iloc[i]
        c = df_1d['close'].iloc[i]
        camarilla_data.append(calculate_camarilla(h, l, c))
    
    # Unpack Camarilla levels
    R4_1d = [x[0] for x in camarilla_data]
    R3_1d = [x[1] for x in camarilla_data]
    R2_1d = [x[2] for x in camarilla_data]
    R1_1d = [x[3] for x in camarilla_data]
    PP_1d = [x[4] for x in camarilla_data]
    S1_1d = [x[5] for x in camarilla_data]
    S2_1d = [x[6] for x in camarilla_data]
    S3_1d = [x[7] for x in camarilla_data]
    S4_1d = [x[8] for x in camarilla_data]
    
    # Align Camarilla levels to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, np.array(R3_1d))
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, np.array(S3_1d))
    R4_1d_aligned = align_htf_to_ltf(prices, df_1d, np.array(R4_1d))
    S4_1d_aligned = align_htf_to_ltf(prices, df_1d, np.array(S4_1d))
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # need EMA34 and volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 1d close > EMA34 = uptrend, close < EMA34 = downtrend
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        trend_up = close_1d_aligned[i] > ema_34_1d_aligned[i]
        trend_down = close_1d_aligned[i] < ema_34_1d_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Camarilla breakout conditions
        breakout_R3 = close[i] > R3_1d_aligned[i]
        breakdown_S3 = close[i] < S3_1d_aligned[i]
        touch_R4 = abs(close[i] - R4_1d_aligned[i]) < (R4_1d_aligned[i] * 0.001)  # within 0.1% of R4
        touch_S4 = abs(close[i] - S4_1d_aligned[i]) < (S4_1d_aligned[i] * 0.001)   # within 0.1% of S4
        touch_ema = abs(close[i] - ema_34_1d_aligned[i]) < (ema_34_1d_aligned[i] * 0.001)  # within 0.1% of EMA
        
        if position == 0:
            # Long: Price breaks above R3 AND uptrend AND volume spike
            if breakout_R3 and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND downtrend AND volume spike
            elif breakdown_S3 and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Price touches R4/S4 or returns to EMA34
            exit_signal = False
            
            if position == 1:
                # Long exit: Price touches R4 or returns to EMA34
                if touch_R4 or touch_ema:
                    exit_signal = True
            elif position == -1:
                # Short exit: Price touches S4 or returns to EMA34
                if touch_S4 or touch_ema:
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