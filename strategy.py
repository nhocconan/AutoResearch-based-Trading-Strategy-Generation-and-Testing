#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike
# Hypothesis: On 12h timeframe, use weekly Camarilla pivot levels (R1/S1) for breakout entries with 1w EMA50 trend filter and volume spike confirmation.
# Enter long when price closes above R1 with volume > 2.0x 20-bar average and 1w EMA50 uptrend.
# Enter short when price closes below S1 with volume > 2.0x 20-bar average and 1w EMA50 downtrend.
# Exit when price crosses the 1w EMA50 (trend reversal) to avoid whipsaw.
# Targets 15-25 trades/year to minimize fee decay while capturing major moves in both bull and bear markets.
# Position sizing: 0.25 for standard conditions, increases to 0.30 when volume > 3.0x average.

name = "12h_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Camarilla pivot calculation and EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
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
    
    # 1-week EMA50 for trend filter
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: dynamic thresholds
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 80  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1w_trend = ema50_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # LONG: Price closes above R1 with volume spike and 1w uptrend
            if close[i] > r1_val and close[i] > ema1w_trend and vol_ratio_val > 2.0:
                # Slight increase in position on extreme volume
                if vol_ratio_val > 3.0:
                    signals[i] = 0.30
                else:
                    signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with volume spike and 1w downtrend
            elif close[i] < s1_val and close[i] < ema1w_trend and vol_ratio_val > 2.0:
                # Slight increase in position on extreme volume
                if vol_ratio_val > 3.0:
                    signals[i] = -0.30
                else:
                    signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1w EMA50 (trend reversal)
            if close[i] < ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if vol_ratio_val <= 3.0 else 0.30
        elif position == -1:
            # EXIT SHORT: Price crosses above 1w EMA50 (trend reversal)
            if close[i] > ema1w_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 if vol_ratio_val <= 3.0 else -0.30
    
    return signals