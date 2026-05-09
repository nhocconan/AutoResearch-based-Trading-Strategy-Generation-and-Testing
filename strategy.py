#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_R1_S1_Breakout_Trend_Volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and volume filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Previous week's close for Camarilla calculation
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Calculate weekly Camarilla levels (R1, S1)
    r1 = prev_close + 1.1 * (prev_high - prev_low) / 4
    s1 = prev_close - 1.1 * (prev_high - prev_low) / 4
    
    # Trend filter: weekly EMA34
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume filter: current weekly volume > 1.5 * 20-week average
    vol_series = pd.Series(df_1w['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1w = df_1w['volume'].values > (vol_ma * 1.5)
    
    # Align all to daily
    r1_daily = align_htf_to_ltf(prices, df_1w, r1)
    s1_daily = align_htf_to_ltf(prices, df_1w, s1)
    ema34_1w_daily = align_htf_to_ltf(prices, df_1w, ema34_1w)
    volume_filter_daily = align_htf_to_ltf(prices, df_1w, volume_filter_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Need enough data for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or
            np.isnan(ema34_1w_daily[i]) or np.isnan(volume_filter_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r1_val = r1_daily[i]
        s1_val = s1_daily[i]
        trend = ema34_1w_daily[i]
        vol_filter = volume_filter_daily[i]
        
        if position == 0:
            # Enter long: break above R1 with volume and above trend
            if close[i] > r1_val and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below S1 with volume and below trend
            elif close[i] < s1_val and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 (mean reversion to center)
            if close[i] < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 (mean reversion to center)
            if close[i] > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals