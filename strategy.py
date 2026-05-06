#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA trend filter and volume confirmation
# Long when 1d Williams %R crosses above -80 from below AND 1d EMA34 > EMA89 AND volume > 1.3 * avg_volume(20)
# Short when 1d Williams %R crosses below -20 from above AND 1d EMA34 < EMA89 AND volume > 1.3 * avg_volume(20)
# Exit when Williams %R crosses -50 in opposite direction or volume drops below average
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies overbought/oversold conditions with reversal potential
# 1d EMA filter ensures alignment with daily trend, reducing counter-trend trades
# Volume confirmation filters weak signals
# Works in bull (buying oversold dips in uptrend) and bear (selling overbought rallies in downtrend)

name = "6h_1dWilliamsR_EMATrend_Volume"
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
    
    # Get 1d data ONCE before loop for Williams %R and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 90:  # Need sufficient data for Williams %R(14) and EMA89
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_series_1d = pd.Series(high_1d)
    low_series_1d = pd.Series(low_1d)
    close_series_1d = pd.Series(close_1d)
    highest_high_14 = high_series_1d.rolling(window=14, min_periods=14).max()
    lowest_low_14 = low_series_1d.rolling(window=14, min_periods=14).min()
    williams_r = (highest_high_14 - close_series_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).values
    
    # Calculate 1d EMA34 and EMA89
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 1.3 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below with EMA34 > EMA89 and volume confirmation
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above with EMA34 < EMA89 and volume confirmation
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 or volume drops below average
            if williams_r_aligned[i] < -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 or volume drops below average
            if williams_r_aligned[i] > -50 or not volume_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals