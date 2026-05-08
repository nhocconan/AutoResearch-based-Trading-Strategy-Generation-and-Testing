#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1-day EMA50 trend filter
# Long when price breaks above upper Donchian band with volume > 1.5x 20-period average and 1d EMA50 upward
# Short when price breaks below lower Donchian band with volume > 1.5x 20-period average and 1d EMA50 downward
# Exit when price crosses opposite Donchian band or 1d trend reverses
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 20-50 trades per year for optimal fee drag (< 200 total over 4 years)
# Donchian provides clear structure, volume confirms momentum, EMA50 filters counter-trend noise

name = "4h_Donchian20_Volume_1dEMA50_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_slope = ema50_1d[1:] - ema50_1d[:-1]  # slope: positive = uptrend
    ema50_1d_slope = np.concatenate([[0], ema50_1d_slope])  # align length
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema50_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d_slope)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_1d_slope_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_val = close[i]
        upper_val = upper_donchian[i]
        lower_val = lower_donchian[i]
        ema50_val = ema50_1d_aligned[i]
        ema50_slope = ema50_1d_slope_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: break above upper Donchian, volume confirmation, 1d uptrend (positive slope)
            if close_val > upper_val and vol_conf_val and ema50_slope > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian, volume confirmation, 1d downtrend (negative slope)
            elif close_val < lower_val and vol_conf_val and ema50_slope < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below lower Donchian or 1d trend turns down
            if close_val < lower_val or ema50_slope < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above upper Donchian or 1d trend turns up
            if close_val > upper_val or ema50_slope > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals