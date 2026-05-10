#!/usr/bin/env python3
"""
1H_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Uses 1h timeframe with 4h timeframe for trend confirmation. Enters long when price breaks above 4h R1 in uptrend (close > EMA34) with volume > 2x 20-period average. Enters short when price breaks below 4h S1 in downtrend (close < EMA34) with volume confirmation. Exits when price returns to opposite level (S1 for long, R1 for short) or trend reverses. Uses 4h EMA34 for trend to avoid whipsaws and works in both bull/bear markets. Targets 15-37 trades per year on 1h timeframe with position size 0.20 to minimize fee drag.
"""

name = "1H_Camarilla_R1_S1_Breakout_4hTrend_Volume"
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
    
    # Get 4h data for Camarilla pivots and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(34) for trend direction
    ema_34_4h = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    pivot_range = prev_high - prev_low
    r1_level = prev_close + 1.1 * pivot_range / 12
    s1_level = prev_close - 1.1 * pivot_range / 12
    
    # Align pivot levels to 1h timeframe (available after 4h bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_level)
    
    # Volume filter: volume > 2x 20-period average on 1h chart
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 2.0
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA34
        price_above_ema = close[i] > ema_34_4h_aligned[i]
        price_below_ema = close[i] < ema_34_4h_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above R1 in uptrend with volume spike
            if (close[i] > r1_aligned[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below S1 in downtrend with volume spike
            elif (close[i] < s1_aligned[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to S1 or trend reverses to downtrend
            if (close[i] < s1_aligned[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to R1 or trend reverses to uptrend
            if (close[i] > r1_aligned[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals