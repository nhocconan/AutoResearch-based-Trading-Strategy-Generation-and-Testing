#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla H3/L3 breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H3 AND 1w EMA200 trend up (price > EMA) AND volume > 1.8 * avg_volume(20) on 4h
# Short when price breaks below 1d Camarilla L3 AND 1w EMA200 trend down (price < EMA) AND volume > 1.8 * avg_volume(20) on 4h
# Exit when price crosses back through the 1d Camarilla midpoint (H3/L3 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 80-160 total trades over 4 years (20-40/year) for 4h timeframe
# Camarilla H3/L3 levels provide moderate breakout structure that works in both trending and ranging markets
# 1w EMA200 filter ensures we trade with the major trend, reducing false breakouts in counter-trend moves
# Volume confirmation (1.8x) validates breakout strength while avoiding excessive trading

name = "4h_1dCamarillaH3L3_1wEMA200_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:  # Need at least one completed daily bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H3, L3, midpoint)
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_low_1d = high_1d - low_1d
    camarilla_h3_1d = close_1d + 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_l3_1d = close_1d - 1.1 * high_low_1d * 1.1 / 4.0
    camarilla_mid_1d = (camarilla_h3_1d + camarilla_l3_1d) / 2.0
    
    # Align 1d Camarilla to 4h timeframe (wait for completed daily bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 1w data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need sufficient data for EMA200 calculation
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend: price above/below EMA200
        # We need the 1w EMA value aligned to current 4h bar, but we also need the 1w close price
        # Get the aligned 1w close price for trend determination
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        if np.isnan(close_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        trend_up = close_1w_aligned[i] > ema_200_1w_aligned[i]
        trend_down = close_1w_aligned[i] < ema_200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Camarilla H3, 1w trend up, volume confirmation, in session
            if (close[i] > camarilla_h3_aligned[i] and 
                trend_up and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla L3, 1w trend down, volume confirmation, in session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  trend_down and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals