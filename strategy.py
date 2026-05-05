#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla H3/L3 breakout with 12h EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H3 AND 12h EMA34 > EMA50 (bullish trend) AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below 1d Camarilla L3 AND 12h EMA34 < EMA50 (bearish trend) AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back through the 1d Camarilla midpoint (H3/L3 average)
# Uses discrete sizing 0.25 to minimize fee churn and manage drawdown
# Target: 60-120 total trades over 4 years (15-30/year) for 12h timeframe
# Camarilla H3/L3 levels provide strong breakout structure with fewer false signals than inner levels
# 12h EMA34/EMA50 crossover ensures we trade with the intermediate-term trend, reducing whipsaw
# Volume confirmation (1.5x) validates breakout strength while avoiding overtrading

name = "12h_1dCamarillaH3L3_12hEMA34EMA50_VolumeConfirm"
timeframe = "12h"
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
    if len(df_1d) < 2:  # Need at least one completed 1d bar
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
    
    # Align 1d Camarilla to 12h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 12h data ONCE before loop for EMA34/EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need at least 50 completed 12h bars for EMA50
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 and EMA50
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 12h timeframe (no additional delay needed for EMA)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
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
            np.isnan(ema34_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla H3, 12h EMA34 > EMA50 (bullish trend), volume confirmation, in session
            if (close[i] > camarilla_h3_aligned[i] and 
                ema34_12h_aligned[i] > ema50_12h_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla L3, 12h EMA34 < EMA50 (bearish trend), volume confirmation, in session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  ema34_12h_aligned[i] < ema50_12h_aligned[i] and 
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