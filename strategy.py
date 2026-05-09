#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_Volume"
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
    
    # Get 1w data for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 for trend
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1d Camarilla levels (R1, S1)
    high_1d = high
    low_1d = low
    close_1d = close
    range_1d = high_1d - low_1d
    camarilla_high = close_1d + 1.1 * range_1d / 12  # R1 level
    camarilla_low = close_1d - 1.1 * range_1d / 12   # S1 level
    
    # 1d volume average for volume filter
    vol_avg_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_1d[i]) or np.isnan(camarilla_high[i]) or 
            np.isnan(camarilla_low[i]) or np.isnan(vol_avg_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema34_1w_1d[i]
        resistance = camarilla_high[i]
        support = camarilla_low[i]
        vol_avg = vol_avg_1d[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above R1 with volume and above 1w EMA34
            if close[i] > resistance and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume and below 1w EMA34
            elif close[i] < support and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below S1 or trend reversal
            if close[i] < support or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above R1 or trend reversal
            if close[i] > resistance or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals