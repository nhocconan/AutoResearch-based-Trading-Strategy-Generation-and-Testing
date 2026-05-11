#!/usr/bin/env python3
name = "1d_Camarilla_R1_S1_Breakout_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

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
    
    # Calculate 1w EMA50 for trend filter (HTF) - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily high/low/close for Camarilla levels
    high_1d = df_1w['high'].values  # Note: df_1w contains weekly data
    low_1d = df_1w['low'].values
    close_1d = df_1w['close'].values
    
    # Camarilla levels: R1, S1 (tighter levels for fewer trades)
    hl_range = high_1d - low_1d
    r1 = close_1d + hl_range * 1.0833  # R1 level
    s1 = close_1d - hl_range * 1.0833  # S1 level
    
    # Align Camarilla levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume filter: 50-period EMA for higher threshold
    vol_ema50 = pd.Series(volume).ewm(span=50, min_periods=50, adjust=False).mean().values
    volume_ok = volume > vol_ema50 * 2.0  # Moderate threshold to balance trade frequency
    
    # Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1w = close[i] > ema50_1w_aligned[i]
        price_below_ema1w = close[i] < ema50_1w_aligned[i]
        breakout_long = close[i] > r1_aligned[i]
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 + above 1w EMA50 + volume spike
            if breakout_long and price_above_ema1w and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S1 + below 1w EMA50 + volume spike
            elif breakout_short and price_below_ema1w and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S1 OR trend reverses
                if close[i] < s1_aligned[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R1 OR trend reverses
                if close[i] > r1_aligned[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals