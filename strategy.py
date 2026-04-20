#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_Camarilla_R1S1_Breakout_Volume_Control"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1w: Calculate weekly trend (close above/below 20-week EMA) ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w = ema_20_1w  # Above = uptrend, below = downtrend
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d: Calculate daily ATR for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d - low_1d, np.abs(high_1d - np.roll(close_1d, 1)))
    tr2 = np.maximum(tr1, np.abs(low_1d - np.roll(close_1d, 1)))
    tr2[0] = np.nan  # First value has no previous close
    atr_1d = pd.Series(tr2).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 1d: Calculate Camarilla pivot levels (using previous day's data) ===
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Get values
        close_val = close[i]
        weekly_trend = trend_1w_aligned[i]
        atr_val = atr_1d_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(weekly_trend) or np.isnan(atr_val) or np.isnan(r1_level) or 
            np.isnan(s1_level) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period average
        atr_ma50 = pd.Series(atr_1d_aligned).rolling(window=50, min_periods=50).mean().values[i]
        if np.isnan(atr_ma50) or atr_val < atr_ma50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and weekly uptrend
            if (close_val > r1_level and   # Break above R1
                vol_ratio_val > 2.0 and    # Strong volume confirmation
                close_val > weekly_trend): # Price above weekly EMA (uptrend)
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and weekly downtrend
            elif (close_val < s1_level and   # Break below S1
                  vol_ratio_val > 2.0 and    # Strong volume confirmation
                  close_val < weekly_trend): # Price below weekly EMA (downtrend)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price drops back below R1 or weekly trend turns down
            if close_val < r1_level or close_val < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price rises back above S1 or weekly trend turns up
            if close_val > s1_level or close_val > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals