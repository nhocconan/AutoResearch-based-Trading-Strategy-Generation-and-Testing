#!/usr/bin/env python3
"""
4h Camarilla R3 S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) from 1d timeframe act as strong support/resistance.
Breakouts above R3 or below S3 with volume confirmation and aligned with 1d EMA34 trend capture
strong momentum moves. Designed for 4h timeframe with tight entry conditions to achieve
19-50 trades/year. Works in bull (breakouts above R3 in uptrend) and bear
(breakouts below S3 in downtrend). Uses discrete position sizing (0.25) to minimize fee drag.
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
    
    # Get 1d data for Camarilla pivots and EMA (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R4, R3, R2, R1, PP, S1, S2, S3, S4
    # PP = (H + L + C) / 3
    # R1 = PP + (H - L) * 1.1 / 12
    # R2 = PP + (H - L) * 1.1 / 6
    # R3 = PP + (H - L) * 1.1 / 4
    # S1 = PP - (H - L) * 1.1 / 12
    # S2 = PP - (H - L) * 1.1 / 6
    # S3 = PP - (H - L) * 1.1 / 4
    pp = (high_1d + low_1d + close_1d) / 3.0
    r3 = pp + (high_1d - low_1d) * 1.1 / 4.0
    s3 = pp - (high_1d - low_1d) * 1.1 / 4.0
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate EMA34 on 1d close for trend
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND volume spike AND price > EMA (uptrend)
            long_entry = (curr_high > r3_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S3 AND volume spike AND price < EMA (downtrend)
            short_entry = (curr_low < s3_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below R3 OR price crosses below EMA (trend change)
            if (curr_low < r3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above S3 OR price crosses above EMA (trend change)
            if (curr_high > s3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0