#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA21 trend filter
# Long when price breaks above upper Donchian band with volume > 1.5x 20-period avg and 12h EMA21 upward
# Short when price breaks below lower Donchian band with volume > 1.5x 20-period avg and 12h EMA21 downward
# Exit when price crosses opposite Donchian band or trend reverses
# Uses 12h timeframe for trend filter (more robust than 1d for 4h trading) and volume confirmation
# Targets 20-40 trades per year for optimal fee drag (< 160 total over 4 years)
# Donchian provides clear structure, volume confirms momentum, 12h EMA21 filters counter-trend noise

name = "4h_Donchian20_Volume_12hEMA21_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_donchian = rolling_max(high, 20)
    lower_donchian = rolling_min(low, 20)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate EMA21 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_12h_slope = ema21_12h[1:] - ema21_12h[:-1]  # slope: positive = uptrend
    ema21_12h_slope = np.concatenate([[0], ema21_12h_slope])  # align length
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    ema21_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema21_12h_aligned[i]) or np.isnan(ema21_12h_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        upper_val = upper_donchian[i]
        lower_val = lower_donchian[i]
        ema21_val = ema21_12h_aligned[i]
        ema21_slope = ema21_12h_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above upper Donchian, volume confirmation, 12h uptrend (positive slope)
            if close_val > upper_val and vol_conf_val and ema21_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian, volume confirmation, 12h downtrend (negative slope)
            elif close_val < lower_val and vol_conf_val and ema21_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below lower Donchian or 12h trend turns down
            if close_val < lower_val or ema21_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above upper Donchian or 12h trend turns up
            if close_val > upper_val or ema21_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals