# 12h_1D_CAMARILLA_R1S1_BREAKOUT_VOLUME
# Hypothesis: 12h breakout from daily Camarilla R1/S1 levels with daily trend filter and volume confirmation.
# Works in bull markets (breakouts with trend) and bear markets (fades from pivot levels).
# Designed for 12-37 trades/year to avoid fee drag on 12h timeframe.
# Uses daily data (1h) for pivot calculation and trend filter, avoiding look-ahead with proper alignment.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1D_CAMARILLA_R1S1_BREAKOUT_VOLUME"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and EMA trend
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema34_daily = pd.Series(df_daily['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Daily Camarilla pivot levels (using previous day's OHLC)
    daily_high = df_daily['high'].shift(1).values
    daily_low = df_daily['low'].shift(1).values
    daily_close = df_daily['close'].shift(1).values
    
    # Camarilla calculations
    range_hl = daily_high - daily_low
    camarilla_pivot = (daily_high + daily_low + daily_close) / 3.0
    camarilla_r1 = camarilla_pivot + range_hl * 1.1 / 12
    camarilla_s1 = camarilla_pivot - range_hl * 1.1 / 12
    
    # Align Camarilla levels to 12h
    camarilla_pivot_12h = align_htf_to_ltf(prices, df_daily, camarilla_pivot)
    camarilla_r1_12h = align_htf_to_ltf(prices, df_daily, camarilla_r1)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_daily, camarilla_s1)
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_12h[i]) or np.isnan(camarilla_pivot_12h[i]) or np.isnan(camarilla_r1_12h[i]) or 
            np.isnan(camarilla_s1_12h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.8 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 1.8
        
        if position == 0:
            # Long: Break above R1 with daily uptrend and volume spike
            if close[i] > camarilla_r1_12h[i] and close[i] > ema34_12h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with daily downtrend and volume spike
            elif close[i] < camarilla_s1_12h[i] and close[i] < ema34_12h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls back below pivot OR daily trend turns down
            if close[i] < camarilla_pivot_12h[i] or close[i] < ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises back above pivot OR daily trend turns up
            if close[i] > camarilla_pivot_12h[i] or close[i] > ema34_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals