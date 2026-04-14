#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams %R with 1-day EMA200 filter and volume confirmation
# Long when Williams %R crosses above -20 from below (oversold reversal) with price above 1-day EMA200 and volume >1.5x 14-period average
# Short when Williams %R crosses below -80 from above (overbought reversal) with price below 1-day EMA200 and volume >1.5x 14-period average
# Exit when Williams %R crosses back to neutral zone (-50) or when price crosses the 1-day EMA200 in opposite direction
# Williams %R is effective at identifying reversal points in ranging markets, while EMA200 filters for higher timeframe trend
# Volume confirmation reduces false signals
# Target: 20-50 total trades over 4 years to minimize fee drag while capturing meaningful reversals

def generate_signals(prices):
    n = len(prices)
    if n < 30:
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
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_4h) / (highest_high - lowest_low) * -100
    
    # Calculate 1-day EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 4h volume average (14-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=14, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for 14-period calculations and EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]  # Current 4h volume
        
        if position == 0:
            # Long setup: Williams %R crosses above -20 from below with volume confirmation and price above 1d EMA200
            if (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and  # Cross above -20
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and                    # Volume confirmation
                price > ema_200_1d_aligned[i]):                                   # Price above 1d EMA200 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: Williams %R crosses below -80 from above with volume confirmation and price below 1d EMA200
            elif (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and  # Cross below -80
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and                     # Volume confirmation
                  price < ema_200_1d_aligned[i]):                                     # Price below 1d EMA200 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses back below -50 or price crosses below 1d EMA200
            if (williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50) or \
               (price < ema_200_1d_aligned[i] and close[i-1] >= ema_200_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses back above -50 or price crosses above 1d EMA200
            if (williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50) or \
               (price > ema_200_1d_aligned[i] and close[i-1] <= ema_200_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0