#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_AngleOfAttack_Donchian_Slope"
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
    
    # Get 1d data for Donchian and slope
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Daily Donchian channels (20 period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper and lower bands
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Donchian width (normalized by price)
    middle_20 = (upper_20 + lower_20) / 2
    width_20 = upper_20 - lower_20
    width_pct = width_20 / middle_20
    
    # Slope of the middle line (rate of change over 5 periods)
    slope_5 = pd.Series(middle_20).diff(5).values / 5  # 5-day change per day
    
    # Align all to 6h
    upper_20_6h = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_6h = align_htf_to_ltf(prices, df_1d, lower_20)
    width_pct_6h = align_htf_to_ltf(prices, df_1d, width_pct)
    slope_5_6h = align_htf_to_ltf(prices, df_1d, slope_5)
    
    # 6-period RSI for overbought/oversold on 6h
    delta = pd.Series(close).diff().values
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/6, adjust=False, min_periods=6).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60  # Ensure all indicators are valid
    
    for i in range(start_idx, n):
        if (np.isnan(upper_20_6h[i]) or np.isnan(lower_20_6h[i]) or 
            np.isnan(width_pct_6h[i]) or np.isnan(slope_5_6h[i]) or 
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: price breaks above upper Donchian + positive slope + not overbought + volume
            if (close[i] > upper_20_6h[i] and 
                slope_5_6h[i] > 0 and 
                rsi[i] < 70 and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + negative slope + not oversold + volume
            elif (close[i] < lower_20_6h[i] and 
                  slope_5_6h[i] < 0 and 
                  rsi[i] > 30 and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below lower Donchian or slope turns negative
            if close[i] < lower_20_6h[i] or slope_5_6h[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian or slope turns positive
            if close[i] > upper_20_6h[i] or slope_5_6h[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals