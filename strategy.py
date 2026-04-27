#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm
Hypothesis: Camarilla R1/S1 levels from 1d chart act as intraday support/resistance. 
Breakout above R1 with volume confirmation and 1w uptrend (price > EMA50) goes long.
Breakdown below S1 with volume confirmation and 1w downtrend (price < EMA50) goes short.
Exits on opposite Camarilla level touch or trend reversal. Uses discrete sizing (0.25) 
to limit fee churn. Designed for 1d timeframe targeting 30-100 total trades over 4 years 
(7-25/year). Works in bull markets via upside breakouts and bear markets via 
downside breakdowns, with volume filter preventing false breakouts in low-liquidity periods.
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
    
    # Get 1d data for Camarilla levels and 1w data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d Camarilla levels: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    rang = high_1d - low_1d
    R1 = close_1d + rang * 1.1 / 2
    S1 = close_1d - rang * 1.1 / 2
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to primary timeframe (1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (tighter to reduce trades)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50 (50), volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        ema_1w_val = ema_50_1w_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine 1w trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_1w_val
            is_downtrend = close_val < ema_1w_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above R1 and volume confirms
                if (close_val > R1_val) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below S1 and volume confirms
                if (close_val < S1_val) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches S1 (support) or trend changes to downtrend
            exit_condition = (close_val < S1_val) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches R1 (resistance) or trend changes to uptrend
            exit_condition = (close_val > R1_val) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0