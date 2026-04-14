#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with 1d Trend Filter and Volume Confirmation
# Uses Camarilla pivot levels (S1/S2/R1/R2) from 12h for breakout entries
# 1d EMA (20) provides trend direction filter to avoid counter-trend trades
# Volume confirmation (>1.8x average) ensures institutional participation
# Designed to work in both bull and bear markets by trading breakouts in direction of 1d trend
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for Camarilla pivots
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels for 12h
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            R2 = C + ((H-L)*1.1/6), R1 = C + ((H-L)*1.1/12)
    #            S1 = C - ((H-L)*1.1/12), S2 = C - ((H-L)*1.1/6)
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot points (using typical price)
    typical_price_12h = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r1_12h = typical_price_12h + (range_12h * 1.1 / 12)
    r2_12h = typical_price_12h + (range_12h * 1.1 / 6)
    s1_12h = typical_price_12h - (range_12h * 1.1 / 12)
    s2_12h = typical_price_12h - (range_12h * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA (20) for trend direction
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.8x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume average and EMA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(r2_12h_aligned[i]) or 
            np.isnan(s1_12h_aligned[i]) or np.isnan(s2_12h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade in direction of 1d EMA
        above_ema = price > ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume filter and above 1d EMA
            if price > r1_12h_aligned[i] and vol > 1.8 * avg_vol[i] and above_ema:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S1 with volume filter and below 1d EMA
            elif price < s1_12h_aligned[i] and vol > 1.8 * avg_vol[i] and not above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 (reversal) or below 1d EMA
            if price < s1_12h_aligned[i] or price < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1 (reversal) or above 1d EMA
            if price > r1_12h_aligned[i] or price > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_1dEMA_Volume"
timeframe = "12h"
leverage = 1.0