#!/usr/bin/env python3
name = "6h_Camarilla_R3S3_Breakout_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Camarilla levels for each weekly bar
    camarilla_R3 = np.zeros_like(close_1w)
    camarilla_S3 = np.zeros_like(close_1w)
    camarilla_R4 = np.zeros_like(close_1w)
    camarilla_S4 = np.zeros_like(close_1w)
    
    # Typical price for weekly bar
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R3/S3 at ±1.1/6 * range from typical price
    # R4/S4 at ±1.5/6 * range from typical price
    camarilla_R3 = typical_price_1w + (1.1/6) * range_1w
    camarilla_S3 = typical_price_1w - (1.1/6) * range_1w
    camarilla_R4 = typical_price_1w + (1.5/6) * range_1w
    camarilla_S4 = typical_price_1w - (1.5/6) * range_1w
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R3_6h = align_htf_to_ltf(prices, df_1w, camarilla_R3)
    camarilla_S3_6h = align_htf_to_ltf(prices, df_1w, camarilla_S3)
    camarilla_R4_6h = align_htf_to_ltf(prices, df_1w, camarilla_R4)
    camarilla_S4_6h = align_htf_to_ltf(prices, df_1w, camarilla_S4)
    
    # Get daily trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike filter on 6h: current volume > 2.0 x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_R3_6h[i]) or np.isnan(camarilla_S3_6h[i]) or 
            np.isnan(camarilla_R4_6h[i]) or np.isnan(camarilla_S4_6h[i]) or 
            np.isnan(ema34_1d_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 with weekly uptrend + volume spike
            if (close[i] > camarilla_R3_6h[i] and 
                ema34_1d_6h[i] > ema34_1d_6h[i-1] and  # rising daily trend
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with weekly downtrend + volume spike
            elif (close[i] < camarilla_S3_6h[i] and 
                  ema34_1d_6h[i] < ema34_1d_6h[i-1] and  # falling daily trend
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or closes below R3
            if close[i] < camarilla_S3_6h[i] or close[i] < camarilla_R3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or closes above S3
            if close[i] > camarilla_R3_6h[i] or close[i] > camarilla_S3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals