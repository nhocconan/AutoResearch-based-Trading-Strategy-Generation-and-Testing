#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day EMA200 trend filter and volume confirmation
# Long when Williams %R crosses above -20 from below with volume > 1.5x 20-period average and price above 1d EMA200
# Short when Williams %R crosses below -80 from above with volume > 1.5x 20-period average and price below 1d EMA200
# Exit when Williams %R crosses the opposite threshold (-80 for long exit, -20 for short exit)
# Williams %R identifies overbought/oversold conditions, EMA200 provides trend filter
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h Williams %R (14-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    
    # Calculate 1d EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # for 200-period EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]
        
        if position == 0:
            # Long setup: Williams %R crosses above -20 from below with volume confirmation and price above 1d EMA200
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                price > ema_200_1d_aligned[i]):                 # Price above 1d EMA200 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -80 from above with volume confirmation and price below 1d EMA200
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  price < ema_200_1d_aligned[i]):                 # Price below 1d EMA200 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses below -80
            if williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses above -20
            if williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0