#!/usr/bin/env python3
"""
1h Camarilla R1S1 Breakout with 4h EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R1/S1) on 1h chart act as intraday support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and aligned 4h EMA34 trend capture 
intraday momentum moves. The 4h EMA34 ensures we trade with higher timeframe momentum, 
reducing false breakouts. Volume spike confirms participation. Designed for moderate 
trade frequency (15-37/year) on 1h timeframe to work in both bull and bear markets 
via trend following with tight risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA34 trend (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on 4h close for trend
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1h data for Camarilla pivot calculation (R1, S1 levels) using previous 1h bar
    if len(prices) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for each 1h bar: based on previous hour's high, low, close
    # We need to shift to avoid look-ahead: use previous hour's data to calculate current hour's levels
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    prev_close = prices['close'].shift(1).values
    
    # Camarilla formulas for R1/S1:
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA, and to avoid NaN from shift
    start_idx = max(34, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1[i]) or 
            np.isnan(camarilla_s1[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_4h_aligned[i]
        r1_level = camarilla_r1[i]
        s1_level = camarilla_s1[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R1 resistance AND volume spike AND price > 4h EMA34 (uptrend)
            long_entry = (curr_close > r1_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below S1 support AND volume spike AND price < 4h EMA34 (downtrend)
            short_entry = (curr_close < s1_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below S1 support (broken support) OR price crosses below EMA (trend change)
            if (curr_close < s1_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price crosses above R1 resistance (broken resistance) OR price crosses above EMA (trend change)
            if (curr_close > r1_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0