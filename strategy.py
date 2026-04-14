#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla pivot (L3) bounce with 1-day trend filter and volume confirmation
# Long when price bounces off Camarilla L3 level (1d) with volume >1.5x 20-period average and price above 1d EMA200
# Short when price rejects at Camarilla H3 level (1d) with volume >1.5x 20-period average and price below 1d EMA200
# Exit when price reaches Camarilla H4 (for longs) or L4 (for shorts) or closes beyond midpoint
# 1-day EMA200 acts as trend filter to avoid counter-trend trades
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
    
    # Calculate 4h OHLC from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 1d OHLC from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from 1d data (previous day's close)
    camarilla_H4 = 1.1 * close_1d - 0.1 * high_1d
    camarilla_H3 = 1.1 * close_1d - 0.1 * low_1d
    camarilla_L3 = 1.1 * close_1d - 0.1 * high_1d
    camarilla_L4 = 1.1 * close_1d - 0.1 * low_1d
    camarilla_mid = (camarilla_H3 + camarilla_L3) / 2
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 200  # for 200-period EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_H3_aligned[i]) or np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]
        
        if position == 0:
            # Long setup: bounce off L3 with volume confirmation and price above 1d EMA200
            if (price >= camarilla_L3_aligned[i] * 0.999 and price <= camarilla_L3_aligned[i] * 1.001 and  # Near L3
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                price > ema_200_1d_aligned[i]):                 # Price above 1d EMA200 for bullish bias
                position = 1
                signals[i] = position_size
            # Short setup: rejection at H3 with volume confirmation and price below 1d EMA200
            elif (price >= camarilla_H3_aligned[i] * 0.999 and price <= camarilla_H3_aligned[i] * 1.001 and  # Near H3
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume confirmation
                  price < ema_200_1d_aligned[i]):                 # Price below 1d EMA200 for bearish bias
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H4 or closes below midpoint
            if price >= camarilla_H4_aligned[i] * 0.999 or price < camarilla_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L4 or closes above midpoint
            if price <= camarilla_L4_aligned[i] * 1.001 or price > camarilla_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_L3H3_Bounce_1dEMA200"
timeframe = "4h"
leverage = 1.0