#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1d EMA34 for trend
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_high = close_1d + 1.1 * range_1d / 12  # R1 level
    camarilla_low = close_1d - 1.1 * range_1d / 12   # S1 level
    
    # 1d volume average for volume filter
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA20 for higher timeframe trend
    ema20_1w = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all to 12h
    ema34_1d_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    camarilla_high_12h = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_12h = align_htf_to_ltf(prices, df_1d, camarilla_low)
    vol_avg_1d_12h = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    ema20_1w_12h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_12h[i]) or np.isnan(camarilla_high_12h[i]) or 
            np.isnan(camarilla_low_12h[i]) or np.isnan(vol_avg_1d_12h[i]) or 
            np.isnan(ema20_1w_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend_1d = ema34_1d_12h[i]
        trend_1w = ema20_1w_12h[i]
        resistance = camarilla_high_12h[i]
        support = camarilla_low_12h[i]
        vol_avg = vol_avg_1d_12h[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        # Only trade when 1d and 1w trends align
        trend_up = trend_1d > trend_1w
        trend_down = trend_1d < trend_1w
        
        if position == 0:
            # Long: break above R1 with volume and bullish alignment
            if close[i] > resistance and vol_ok and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and bearish alignment
            elif close[i] < support and vol_ok and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 or trend turns bearish
            if close[i] < support or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 or trend turns bullish
            if close[i] > resistance or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals