#!/usr/bin/env python3
# 4H_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as institutional support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and daily trend filter (EMA34) capture
# institutional breakout moves. Works in bull markets (breakouts continue) and bear markets
# (breakdowns continue) by aligning with higher timeframe trend. Volume filter avoids false breakouts.
# Target: 20-50 trades per year (~80-200 over 4 years) with position size 0.25.

name = "4H_Camarilla_R1S1_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # Load daily data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous completed daily bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (available after daily bar close)
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 20-period average volume
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend
        daily_uptrend = close[i] > ema_34_4h[i]
        daily_downtrend = close[i] < ema_34_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 + daily uptrend + volume confirmation
            if close[i] > r1_4h[i] and daily_uptrend and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily downtrend + volume confirmation
            elif close[i] < s1_4h[i] and daily_downtrend and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or trend reversal
            if close[i] < s1_4h[i] or not daily_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or trend reversal
            if close[i] > r1_4h[i] or not daily_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals