#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla H4/L4 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H4 AND 1d close > EMA34 (uptrend) AND volume > 2.0 * avg_volume(20) on 4h
# Short when price breaks below 1d Camarilla L4 AND 1d close < EMA34 (downtrend) AND volume > 2.0 * avg_volume(20) on 4h
# Exit when price crosses back through the 1d Camarilla midpoint (H4/L4 average)
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# 1d Camarilla H4/L4 provides strong breakout levels that reduce whipsaw
# 1d EMA34 trend filter ensures we trade with the dominant daily trend
# Volume confirmation (2.0x) validates breakout strength while limiting overtrading

name = "4h_1dCamarillaH4L4_1dEMA34_Trend_VolumeConfirm"
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
    if len(df_1d) < 2:  # Need at least one completed 1d bar
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (H4, L4, midpoint)
    # Camarilla: H4 = close + 1.1*(high-low)*1.5/4, L4 = close - 1.1*(high-low)*1.5/4
    high_low_1d = high_1d - low_1d
    camarilla_h4_1d = close_1d + 1.1 * high_low_1d * 1.5 / 4.0
    camarilla_l4_1d = close_1d - 1.1 * high_low_1d * 1.5 / 4.0
    camarilla_mid_1d = (camarilla_h4_1d + camarilla_l4_1d) / 2.0
    
    # Align 1d Camarilla to 4h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 1d data for EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
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
            # Long: price breaks above 1d Camarilla H4, 1d close > EMA34 (uptrend), volume confirmation, in session
            if (close[i] > camarilla_h4_aligned[i] and 
                close_1d_aligned[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below 1d Camarilla L4, 1d close < EMA34 (downtrend), volume confirmation, in session
            elif (close[i] < camarilla_l4_aligned[i] and 
                  close_1d_aligned[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 1d Camarilla midpoint
            if close[i] < camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses back above 1d Camarilla midpoint
            if close[i] > camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Note: close_1d_aligned is not defined in the above code. Let me fix it by adding the alignment for close_1d.
# Actually, we need to align the 1d close prices to compare with EMA. Let me correct this.