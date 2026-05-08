#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h price action at 1d Camarilla pivot levels with volume confirmation.
# Long when price bounces off S1/S2 (support) with volume spike AND closes above open.
# Short when price rejects at R1/R2 (resistance) with volume spike AND closes below open.
# Exit when price reaches opposite pivot level (S1 for longs, R1 for shorts) or shows reversal signal.
# Camarilla levels provide high-probability reversal zones. Volume confirms institutional interest.
# Works in both bull/bear markets as it fades extremes rather than following trends.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Camarilla_Pivot_Bounce_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla pivot levels (based on previous day)
    # R4 = close + (high-low)*1.5000
    # R3 = close + (high-low)*1.2500
    # R2 = close + (high-low)*1.1666
    # R1 = close + (high-low)*1.0833
    # PP = (high+low+close)/3
    # S1 = close - (high-low)*1.0833
    # S2 = close - (high-low)*1.1666
    # S3 = close - (high-low)*1.2500
    # S4 = close - (high-low)*1.5000
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    
    camarilla_r1 = camarilla_pp + camarilla_range * 1.0833
    camarilla_r2 = camarilla_pp + camarilla_range * 1.1666
    camarilla_s1 = camarilla_pp - camarilla_range * 1.0833
    camarilla_s2 = camarilla_pp - camarilla_range * 1.1666
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    
    # Volume filter: current volume > 2.0x 20-period average (stricter for fewer trades)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price at/below S2, volume spike, bullish close
            near_support = (low[i] <= camarilla_s2_aligned[i]) or (close[i] <= camarilla_s2_aligned[i] * 1.001)
            bullish_close = close[i] > open_price[i]
            
            if near_support and volume_filter[i] and bullish_close:
                signals[i] = 0.25
                position = 1
            
            # Short conditions: price at/above R2, volume spike, bearish close
            elif (high[i] >= camarilla_r2_aligned[i]) or (close[i] >= camarilla_r2_aligned[i] * 0.999):
                bearish_close = close[i] < open_price[i]
                if volume_filter[i] and bearish_close:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches S1 or shows bearish reversal
            if close[i] >= camarilla_s1_aligned[i] or close[i] < open_price[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches R1 or shows bullish reversal
            if close[i] <= camarilla_r1_aligned[i] or close[i] > open_price[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals