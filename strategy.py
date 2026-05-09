#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 12h EMA50 for trend
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d EMA13 for Elder Ray (standard period)
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align all to 6h
    ema50_12h_6h = align_htf_to_ltf(prices, df_12h, ema50_12h)
    bull_power_1d_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_12h_6h[i]) or np.isnan(bull_power_1d_6h[i]) or 
            np.isnan(bear_power_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_12h_6h[i]
        bull_power = bull_power_1d_6h[i]
        bear_power = bear_power_1d_6h[i]
        
        # Volume filter: current volume > 1.5x 6-period average
        if i >= 6:
            vol_avg = np.mean(volume[i-6:i])
            vol_ok = volume[i] > vol_avg * 1.5
        else:
            vol_ok = False
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) + above trend + volume
            if bull_power > 0 and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling) + below trend + volume
            elif bear_power < 0 and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power turns negative (selling pressure) or trend reversal
            if bear_power < 0 or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power turns positive (buying pressure) or trend reversal
            if bull_power > 0 or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals