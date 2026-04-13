#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot + 12h EMA trend + volume spike.
# Camarilla pivot levels from 12h provide robust support/resistance.
# Long when price bounces from L3 (support) with volume spike and above 12h EMA200 (trend).
# Short when price rejects from H3 (resistance) with volume spike and below 12h EMA200.
# Works in bull/bear by using 12h EMA200 as trend filter and requiring volume confirmation.
# Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Camarilla pivot and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h Camarilla pivot levels
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pp = (high + low + close) / 3
        range_val = high - low
        
        # Camarilla levels
        h4 = pp + range_val * 1.1 / 2
        h3 = pp + range_val * 1.1 / 4
        h2 = pp + range_val * 1.1 / 6
        h1 = pp + range_val * 1.1 / 12
        
        l1 = pp - range_val * 1.1 / 12
        l2 = pp - range_val * 1.1 / 6
        l3 = pp - range_val * 1.1 / 4
        l4 = pp - range_val * 1.1 / 2
        
        return h1, h2, h3, h4, l1, l2, l3, l4
    
    h1_12h, h2_12h, h3_12h, h4_12h, l1_12h, l2_12h, l3_12h, l4_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # 12h EMA200 for trend filter
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 12h indicators to 4h timeframe
    h1_12h_aligned = align_htf_to_ltf(prices, df_12h, h1_12h)
    h2_12h_aligned = align_htf_to_ltf(prices, df_12h, h2_12h)
    h3_12h_aligned = align_htf_to_ltf(prices, df_12h, h3_12h)
    h4_12h_aligned = align_htf_to_ltf(prices, df_12h, h4_12h)
    l1_12h_aligned = align_htf_to_ltf(prices, df_12h, l1_12h)
    l2_12h_aligned = align_htf_to_ltf(prices, df_12h, l2_12h)
    l3_12h_aligned = align_htf_to_ltf(prices, df_12h, l3_12h)
    l4_12h_aligned = align_htf_to_ltf(prices, df_12h, l4_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(h3_12h_aligned[i]) or np.isnan(l3_12h_aligned[i]) or 
            np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = vol_ma[i]
        h3 = h3_12h_aligned[i]
        l3 = l3_12h_aligned[i]
        ema_trend = ema_200_12h_aligned[i]
        
        # Volume spike: current volume > 2.0x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: price near L3 support + volume spike + above EMA200
            if (price <= l3 * 1.005 and  # within 0.5% of L3
                volume_spike and
                price > ema_trend):
                position = 1
                signals[i] = position_size
            # Short: price near H3 resistance + volume spike + below EMA200
            elif (price >= h3 * 0.995 and  # within 0.5% of H3
                  volume_spike and
                  price < ema_trend):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below L3 or below EMA200
            if (price < l3 * 0.995 or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above H3 or above EMA200
            if (price > h3 * 1.005 or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Camarilla_Pivot_EMA_Volume"
timeframe = "4h"
leverage = 1.0