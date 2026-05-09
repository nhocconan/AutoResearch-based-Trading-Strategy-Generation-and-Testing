#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Supertrend_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Supertrend on 6h data
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First element has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[atr_period] = np.mean(tr[:atr_period+1])
    for i in range(atr_period+1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Final Supertrend bands
    final_upper = np.full_like(close, np.nan)
    final_lower = np.full_like(close, np.nan)
    supertrend = np.full_like(close, np.nan)
    trend_dir = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, len(close)):
        if i == atr_period:
            final_upper[i] = upper_band[i]
            final_lower[i] = lower_band[i]
        else:
            final_upper[i] = upper_band[i] if (upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]) else final_upper[i-1]
            final_lower[i] = lower_band[i] if (lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]) else final_lower[i-1]
        
        if i == atr_period:
            supertrend[i] = final_lower[i]
            trend_dir[i] = 1
        else:
            if supertrend[i-1] == final_upper[i-1]:
                supertrend[i] = final_lower[i] if close[i] > final_lower[i] else final_upper[i]
            else:
                supertrend[i] = final_upper[i] if close[i] < final_upper[i] else final_lower[i]
            
            trend_dir[i] = 1 if supertrend[i] == final_lower[i] else -1
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 6h volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(atr_period, 50, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend[i]) or np.isnan(trend_dir[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        st = supertrend[i]
        td = trend_dir[i]
        trend = ema50_1d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Supertrend uptrend + price above 1d EMA50 + volume filter
            if td == 1 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Supertrend downtrend + price below 1d EMA50 + volume filter
            elif td == -1 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend turns down
            if td == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Supertrend turns up
            if td == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#%%