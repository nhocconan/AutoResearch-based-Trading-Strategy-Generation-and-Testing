#/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
# Hypothesis: On 12h timeframe, enter long when price closes above weekly R1 with close > weekly EMA34 and volume > 2x average.
# Enter short when price closes below weekly S1 with close < weekly EMA34 and volume > 2x average.
# Exit when price crosses weekly EMA34 (trend reversal).
# Uses weekly trend filter for long-term bias, works in bull markets via breakouts and in bear via short reversals at S1.
# Targets 15-30 trades/year for low fee drag.

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
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
    
    # Load weekly data for Camarilla pivot calculation and EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate pivot point and range
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Camarilla R1 and S1 levels
    r1 = weekly_pivot + weekly_range * 1.083
    s1 = weekly_pivot - weekly_range * 1.083
    
    # 1-week EMA34 for trend filter
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1w_trend = ema34_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price closes above R1 with volume > 2x average and close > weekly EMA34
            if close[i] > r1_val and close[i] > ema1w_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with volume > 2x average and close < weekly EMA34
            elif close[i] < s1_val and close[i] < ema1w_trend and volume[i] > 2.0 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below weekly EMA34 (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly EMA34 (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals