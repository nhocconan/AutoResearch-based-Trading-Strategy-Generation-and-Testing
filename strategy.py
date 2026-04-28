#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Camarilla pivot levels (R3/S3) breakout with volume confirmation and 1d EMA trend filter.
# Enter long when price breaks above R3 with volume > 2.0x average and close > 1d EMA34 (bullish bias).
# Enter short when price breaks below S3 with volume > 2.0x average and close < 1d EMA34 (bearish bias).
# Exit when price returns to the 12h pivot level (PP) or opposite Camarilla level is touched.
# Combines Camarilla breakout logic with higher timeframe trend filter to reduce whipsaws.
# Works in bull markets (breakouts continue up with trend) and bear markets (breakdowns continue down with trend).
# Uses discrete position sizing (0.25) to control risk. Target: 50-150 total trades over 4 years.

name = "6h_Camarilla_R3S3_Breakout_12hVolume2x_1dEMA34_v1"
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
    
    # Get 12h data for Camarilla pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Pivot Point (PP)
    PP = (high_12h + low_12h + close_12h) / 3.0
    # Range
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    R3 = PP + range_12h * 1.1 / 4.0
    S3 = PP - range_12h * 1.1 / 4.0
    R4 = PP + range_12h * 1.1 / 2.0
    S4 = PP - range_12h * 1.1 / 2.0
    PP_level = PP  # for exit
    
    # Align Camarilla levels to 6h timeframe
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    R4_aligned = align_htf_to_ltf(prices, df_12h, R4)
    S4_aligned = align_htf_to_ltf(prices, df_12h, S4)
    PP_level_aligned = align_htf_to_ltf(prices, df_12h, PP_level)
    
    # Get 1d data for EMA34 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 1d EMA34 bias
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > R3_aligned[i]
        short_breakout = close[i] < S3_aligned[i]
        
        # Exit conditions: return to pivot level (PP)
        long_exit = close[i] < PP_level_aligned[i]
        short_exit = close[i] > PP_level_aligned[i]
        
        # Entry conditions
        long_entry = long_breakout and vol_confirm and bullish_bias
        short_entry = short_breakout and vol_confirm and bearish_bias
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals