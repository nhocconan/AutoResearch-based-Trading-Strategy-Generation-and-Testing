#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla H4 AND 1d EMA34 > EMA34 previous (uptrend) AND volume > 1.8 * avg_volume(20) on 6h
# Short when price breaks below 1w Camarilla L4 AND 1d EMA34 < EMA34 previous (downtrend) AND volume > 1.8 * avg_volume(20) on 6h
# Exit when price crosses back through the 1w Camarilla midpoint (H4/L4 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Weekly Camarilla H4/L4 provides strong breakout levels that reduce whipsaw
# 1d EMA34 trend filter ensures we trade with the dominant daily trend
# Higher volume threshold (1.8x) limits overtrading while confirming breakout strength

name = "6h_1wCamarillaH4L4_1dEMA34_Trend_VolumeConfirm"
timeframe = "6h"
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
    
    # Calculate 1w Camarilla levels (H4, L4, midpoint)
    # Camarilla: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    high_low_1w = high_1w - low_1w
    camarilla_h4_1w = close_1w + 1.1 * high_low_1w * 1.1 / 2.0
    camarilla_l4_1w = close_1w - 1.1 * high_low_1w * 1.1 / 2.0
    camarilla_mid_1w = (camarilla_h4_1w + camarilla_l4_1w) / 2.0
    
    # Align 1w Camarilla to 6h timeframe (wait for completed weekly bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4_1w)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4_1w)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1w, camarilla_mid_1w)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.8 * 20-period average volume on 6h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla H4, 1d EMA34 > EMA34 previous (uptrend), volume confirmation, in session
            if (close[i] > camarilla_h4_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla L4, 1d EMA34 < EMA34 previous (downtrend), volume confirmation, in session
            elif (close[i] < camarilla_l4_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1w Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses back above 1w Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals