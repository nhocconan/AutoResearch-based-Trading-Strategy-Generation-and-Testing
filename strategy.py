#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week EMA10 trend filter with 1-day Williams %R reversal signals.
In uptrend (price > 1w EMA10), buy when daily Williams %R crosses above -80 from oversold.
In downtrend (price < 1w EMA10), sell when daily Williams %R crosses below -20 from overbought.
Volume confirmation requires current volume > 1.5x 20-day average to filter weak signals.
Exit on trend reversal or when Williams %R returns to neutral territory (-50).
Designed for 10-25 trades/year (40-100 total over 4 years) to minimize fee flood while capturing mean-reversion within trends.
Works in bull markets via buying oversold dips in uptrends and in bear markets via selling overbought rallies in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 1d timeframe (no additional delay needed as it's based on completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load 1w data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_10 = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_aligned = align_htf_to_ltf(prices, df_1w, ema_10)
    
    # Volume confirmation (volume > 1.5x 20-day average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_10_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_10_aligned[i]
        williams_r_val = williams_r_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: Williams %R crosses above -80 from oversold + uptrend + volume confirmation
            if (williams_r_val > -80 and williams_r_aligned[i-1] <= -80 and
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: Williams %R crosses below -20 from overbought + downtrend + volume confirmation
            elif (williams_r_val < -20 and williams_r_aligned[i-1] >= -20 and
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR Williams %R returns to neutral territory
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < ema_trend:
                exit_signal = True
            elif position == -1 and price_close > ema_trend:
                exit_signal = True
            
            # Williams %R mean reversion exit
            if position == 1 and williams_r_val > -50:
                exit_signal = True
            elif position == -1 and williams_r_val < -50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA10_Volume"
timeframe = "1d"
leverage = 1.0