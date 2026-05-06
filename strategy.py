#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1w Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla H3 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 12h
# Short when price breaks below 1w Camarilla L3 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 12h
# Exit when price crosses back through the 1w Camarilla midpoint (H3/L3 average)
# Uses discrete sizing 0.28 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1w Camarilla H3/L3 provides stronger breakout levels than 1d, reducing whipsaw in bear markets
# 1d EMA34 trend filter ensures we trade with the dominant daily trend while adapting faster than weekly
# Higher volume threshold (1.8x) confirms strong breakout conviction while limiting overtrading

name = "12h_1wCamarillaH3L3_1dEMA34_Trend_VolumeConfirm"
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
    
    # Get 1w data ONCE before loop for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:  # Need at least one completed weekly bar
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (H3, L3, midpoint)
    # Camarilla: H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
    high_low_1w = high_1w - low_1w
    camarilla_h3_1w = close_1w + 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_l3_1w = close_1w - 1.1 * high_low_1w * 1.1 / 4.0
    camarilla_mid_1w = (camarilla_h3_1w + camarilla_l3_1w) / 2.0
    
    # Align 1w Camarilla to 12h timeframe (wait for completed weekly bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3_1w)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3_1w)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1w, camarilla_mid_1w)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 12h
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
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla H3, 1d EMA34 > EMA34 previous (uptrend), volume confirmation, in session
            if (close[i] > camarilla_h3_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.28
                position = 1
            # Short: price breaks below 1w Camarilla L3, 1d EMA34 < EMA34 previous (downtrend), volume confirmation, in session
            elif (close[i] < camarilla_l3_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.28
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1w Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        elif position == -1:
            # Exit short: price crosses back above 1w Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals