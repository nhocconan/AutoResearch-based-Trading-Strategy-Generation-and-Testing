#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with daily volume confirmation and 1d trend filter.
# Camarilla levels provide precise support/resistance for reversals/breakouts.
# Volume filter ensures institutional participation.
# Daily trend filter avoids counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros(len(close_1d))  # Resistance 4
    camarilla_l4 = np.zeros(len(close_1d))  # Support 4
    camarilla_h3 = np.zeros(len(close_1d))  # Resistance 3
    camarilla_l3 = np.zeros(len(close_1d))  # Support 3
    
    for i in range(1, len(close_1d)):
        # Previous day's values
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        range_val = phigh - plow
        
        camarilla_h4[i] = pclose + range_val * 1.1 / 2
        camarilla_l4[i] = pclose - range_val * 1.1 / 2
        camarilla_h3[i] = pclose + range_val * 1.1 / 4
        camarilla_l3[i] = pclose - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Daily trend filter: 20-period EMA
    ema_20_1d = np.zeros(len(close_1d))
    for i in range(20, len(close_1d)):
        if i == 20:
            ema_20_1d[i] = np.mean(close_1d[:20])
        else:
            ema_20_1d[i] = close_1d[i] * 0.1 + ema_20_1d[i-1] * 0.9
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume confirmation: 20-period average volume on 12h timeframe
    avg_volume = np.zeros(n)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        h4 = camarilla_h4_aligned[i]
        l4 = camarilla_l4_aligned[i]
        h3 = camarilla_h3_aligned[i]
        l3 = camarilla_l3_aligned[i]
        ema_20 = ema_20_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.3x average volume
        volume_confirm = vol > 1.3 * avg_vol
        
        if position == 0:
            # Long: price closes above H3 with volume + above daily EMA20
            if (price > h3 and 
                volume_confirm and
                price > ema_20):
                position = 1
                signals[i] = position_size
            # Short: price closes below L3 with volume + below daily EMA20
            elif (price < l3 and 
                  volume_confirm and
                  price < ema_20):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below L4 or volume drops significantly
            if (price < l4 or 
                vol < 0.6 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above H4 or volume drops significantly
            if (price > h4 or 
                vol < 0.6 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Camarilla_Pivot_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0