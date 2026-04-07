#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v2
Hypothesis: Use 1-day Camarilla pivot levels with EMA(21) trend filter and volume confirmation on 6h timeframe.
In bull markets: Buy near S3/S4 with trend up, sell at R3/R4. In bear markets: Sell near R3/R4 with trend down, buy at S3/S4.
Camarilla levels provide intraday support/resistance that works in ranging markets, while EMA filter avoids counter-trend trades.
Volume ensures institutional participation. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA(21) for trend filter on 6h
    close_series = pd.Series(close)
    ema21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # HLC = (High + Low + Close) / 3
    # Camarilla: 
    #   R4 = Close + (High - Low) * 1.1/2
    #   R3 = Close + (High - Low) * 1.1/4
    #   R2 = Close + (High - Low) * 1.1/6
    #   R1 = Close + (High - Low) * 1.1/12
    #   S1 = Close - (High - Low) * 1.1/12
    #   S2 = Close - (High - Low) * 1.1/6
    #   S3 = Close - (High - Low) * 1.1/4
    #   S4 = Close - (High - Low) * 1.1/2
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First day has no previous - set to same day (will be filtered by min_periods later)
    high_1d_prev[0] = high_1d[0]
    low_1d_prev[0] = low_1d[0]
    close_1d_prev[0] = close_1d[0]
    
    # Calculate Camarilla levels
    hlc = (high_1d_prev + low_1d_prev + close_1d_prev) / 3
    range_1d = high_1d_prev - low_1d_prev
    
    r4 = close_1d_prev + range_1d * 1.1 / 2
    r3 = close_1d_prev + range_1d * 1.1 / 4
    r2 = close_1d_prev + range_1d * 1.1 / 6
    r1 = close_1d_prev + range_1d * 1.1 / 12
    s1 = close_1d_prev - range_1d * 1.1 / 12
    s2 = close_1d_prev - range_1d * 1.1 / 6
    s3 = close_1d_prev - range_1d * 1.1 / 4
    s4 = close_1d_prev - range_1d * 1.1 / 2
    
    # Align to 6h timeframe
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):
        # Skip if data not available
        if (np.isnan(ema21[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below EMA21
        price_above_ema = close[i] > ema21[i]
        price_below_ema = close[i] < ema21[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or trend turns bearish
            if close[i] >= r3_6h[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or trend turns bullish
            if close[i] <= s3_6h[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Buy near S3/S4 in uptrend
                if price_above_ema and (close[i] <= s3_6h[i] * 1.005 or close[i] <= s4_6h[i] * 1.005):
                    position = 1
                    signals[i] = 0.25
                # Sell near R3/R4 in downtrend
                elif price_below_ema and (close[i] >= r3_6h[i] * 0.995 or close[i] >= r4_6h[i] * 0.995):
                    position = -1
                    signals[i] = -0.25
    
    return signals