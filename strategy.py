#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla H4/L4 breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1d Camarilla H4 AND 1w EMA34 > EMA34 previous 2 bars (strong uptrend) AND volume > 2.0 * avg_volume(50) on 12h
# Short when price breaks below 1d Camarilla L4 AND 1w EMA34 < EMA34 previous 2 bars (strong downtrend) AND volume > 2.0 * avg_volume(50) on 12h
# Exit when price crosses back through the 1d Camarilla midpoint (H4/L4 average)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Camarilla H4/L4 provides stronger breakout levels than H3/L3 to reduce whipsaw and overtrading
# 1w EMA34 trend filter with 2-bar confirmation ensures we trade with strong weekly momentum
# Volume confirmation (2.0x) validates breakout strength while significantly limiting trades

name = "12h_1dCamarillaH4L4_1wEMA34_Trend_VolumeConfirm"
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
    
    # Calculate 1d Camarilla levels (H4, L4, midpoint)
    # Camarilla: H4 = close + 1.5*(high-low)*1.1/2, L4 = close - 1.5*(high-low)*1.1/2
    high_low_1d = high_1d - low_1d
    camarilla_h4_1d = close_1d + 1.5 * high_low_1d * 1.1 / 2.0
    camarilla_l4_1d = close_1d - 1.5 * high_low_1d * 1.1 / 2.0
    camarilla_mid_1d = (camarilla_h4_1d + camarilla_l4_1d) / 2.0
    
    # Align 1d Camarilla to 12h timeframe (wait for completed 1d bar)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid_1d)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need at least 34 completed weekly bars for EMA34
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 50-period average volume on 12h
    avg_volume_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_confirm = volume > (2.0 * avg_volume_50)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume_50[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1d Camarilla H4, 1w EMA34 > EMA34 previous 2 bars (strong uptrend), volume confirmation, in session
            if (close[i] > camarilla_h4_aligned[i] and 
                ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1] and 
                ema_34_1w_aligned[i-1] > ema_34_1w_aligned[i-2] and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Camarilla L4, 1w EMA34 < EMA34 previous 2 bars (strong downtrend), volume confirmation, in session
            elif (close[i] < camarilla_l4_aligned[i] and 
                  ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1] and 
                  ema_34_1w_aligned[i-1] < ema_34_1w_aligned[i-2] and 
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