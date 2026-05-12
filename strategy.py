#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike
# Hypothesis: On 12h timeframe, use daily Camarilla pivot levels (R1/S1) for breakout entries with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price closes above R1 with volume > 2.0x 20-bar average and 1d EMA34 uptrend.
# Enter short when price closes below S1 with volume > 2.0x 20-bar average and 1d EMA34 downtrend.
# Exit when price crosses the 1d EMA34 (trend reversal) to avoid whipsaw.
# Targets 12-37 trades/year to minimize fee drag while capturing meaningful moves in both bull and bear markets.
# Position sizing: 0.25 for standard conditions, increases to 0.35 when volume > 3.0x average.

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Camarilla pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate pivot point and range
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    # Camarilla R1 and S1 levels
    r1 = daily_pivot + daily_range * 1.083
    s1 = daily_pivot - daily_range * 1.083
    
    # 1-day EMA34 for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: dynamic thresholds
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)  # Avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1d_trend = ema34_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # LONG: Price closes above R1 with volume spike and 1d uptrend
            if close[i] > r1_val and close[i] > ema1d_trend and vol_ratio_val > 2.0:
                # Increase position on extreme volume
                if vol_ratio_val > 3.0:
                    signals[i] = 0.35
                else:
                    signals[i] = 0.25
                position = 1
            # SHORT: Price closes below S1 with volume spike and 1d downtrend
            elif close[i] < s1_val and close[i] < ema1d_trend and vol_ratio_val > 2.0:
                # Increase position on extreme volume
                if vol_ratio_val > 3.0:
                    signals[i] = -0.35
                else:
                    signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA34 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if vol_ratio_val <= 3.0 else 0.35
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA34 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 if vol_ratio_val <= 3.0 else -0.35
    
    return signals