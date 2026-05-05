#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla H3/L3 breakout with 1d EMA200 trend filter and volume confirmation
# Long when price breaks above 4h Camarilla H3 AND close > 1d EMA200 (bullish trend) AND volume > 1.5 * avg_volume(20) on 1h
# Short when price breaks below 4h Camarilla L3 AND close < 1d EMA200 (bearish trend) AND volume > 1.5 * avg_volume(20) on 1h
# Exit when price crosses back through the 4h Camarilla midpoint (H3/L3 average)
# Uses discrete sizing 0.20 to control risk and reduce fee churn
# Target: 80-120 total trades over 4 years (20-30/year) for 1h timeframe
# 4h Camarilla provides structure, 1d EMA200 filters for trend regime, volume confirms breakout strength
# Session filter (08-20 UTC) reduces noise during low-liquidity hours

name = "1h_4hCamarillaH3L3_1dEMA200_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:  # Need at least one completed 4h bar
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla levels (H3, L3, midpoint)
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_low_4h = high_4h - low_4h
    camarilla_h3_4h = close_4h + 1.1 * high_low_4h * 1.1 / 4.0
    camarilla_l3_4h = close_4h - 1.1 * high_low_4h * 1.1 / 4.0
    camarilla_mid_4h = (camarilla_h3_4h + camarilla_l3_4h) / 2.0
    
    # Align 4h Camarilla to 1h timeframe (wait for completed 4h bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid_4h)
    
    # Get 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # Need at least 200 completed daily bars for EMA200
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Camarilla H3, close > 1d EMA200 (bullish trend), volume confirmation, in session
            if (close[i] > camarilla_h3_aligned[i] and 
                close[i] > ema_200_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Camarilla L3, close < 1d EMA200 (bearish trend), volume confirmation, in session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  close[i] < ema_200_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above 4h Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals