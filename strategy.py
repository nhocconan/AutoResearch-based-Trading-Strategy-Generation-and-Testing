#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Champaign_Channel"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 250:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for 52-week high/low
    df_weekly = get_htf_data(prices, '1w')
    
    if len(df_weekly) < 52:
        return np.zeros(n)
    
    # Calculate 52-week high and low
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # 52-week high and low (minimum 52 weeks)
    high_52w = pd.Series(high_weekly).rolling(window=52, min_periods=52).max().values
    low_52w = pd.Series(low_weekly).rolling(window=52, min_periods=52).min().values
    
    # Align weekly 52-week levels to daily
    high_52w_daily = align_htf_to_ltf(prices, df_weekly, high_52w)
    low_52w_daily = align_htf_to_ltf(prices, df_weekly, low_52w)
    
    # Daily 200-day SMA for trend filter
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Volume spike detection (20-day average)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_52w_daily[i]) or np.isnan(low_52w_daily[i]) or 
            np.isnan(sma200[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5 x 20-day average
        vol_spike = volume[i] > vol_avg[i] * 1.5
        
        if position == 0:
            # Long: Near 52-week low with uptrend and volume spike
            if (close[i] <= low_52w_daily[i] * 1.02 and  # Within 2% of 52-week low
                close[i] > sma200[i] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Near 52-week high with downtrend and volume spike
            elif (close[i] >= high_52w_daily[i] * 0.98 and  # Within 2% of 52-week high
                  close[i] < sma200[i] and vol_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price reaches 52-week high OR trend turns down
            if (close[i] >= high_52w_daily[i] * 0.98 or  # Near 52-week high
                close[i] < sma200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price reaches 52-week low OR trend turns up
            if (close[i] <= low_52w_daily[i] * 1.02 or  # Near 52-week low
                close[i] > sma200[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals