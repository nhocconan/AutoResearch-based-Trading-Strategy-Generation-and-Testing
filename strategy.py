# 4h_1d_1w_Camarilla_Pivot_Breakout_v1
# Hypothesis: Use daily and weekly Camarilla pivot levels on 4h timeframe with volume confirmation.
# Long when price breaks above daily R4 or weekly R4 with volume > 1.5x 20-period average.
# Short when price breaks below daily S4 or weekly S4 with volume > 1.5x 20-period average.
# Exit when price returns to daily or weekly pivot level.
# Camarilla levels are known to work well in both trending and ranging markets due to their
# mathematical construction based on previous day's range. Weekly levels add higher timeframe
# context. Volume confirmation reduces false breakouts. Designed for low trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_Camarilla_Pivot_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high_d = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low_d = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    prev_close_d = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    
    # Previous week's OHLC for Camarilla calculation
    prev_high_w = df_1w['high'].iloc[-2] if len(df_1w) >= 2 else df_1w['high'].iloc[-1]
    prev_low_w = df_1w['low'].iloc[-2] if len(df_1w) >= 2 else df_1w['low'].iloc[-1]
    prev_close_w = df_1w['close'].iloc[-2] if len(df_1w) >= 2 else df_1w['close'].iloc[-1]
    
    # Calculate daily Camarilla levels
    range_d = prev_high_d - prev_low_d
    if range_d <= 0:
        return np.zeros(n)
    
    daily_r4 = prev_close_d + range_d * 1.1 / 2  # R4 = Close + 1.1 * Range / 2
    daily_s4 = prev_close_d - range_d * 1.1 / 2  # S4 = Close - 1.1 * Range / 2
    daily_pivot = (prev_high_d + prev_low_d + prev_close_d) / 3
    
    # Calculate weekly Camarilla levels
    range_w = prev_high_w - prev_low_w
    if range_w <= 0:
        return np.zeros(n)
    
    weekly_r4 = prev_close_w + range_w * 1.1 / 2  # R4 = Close + 1.1 * Range / 2
    weekly_s4 = prev_close_w - range_w * 1.1 / 2  # S4 = Close - 1.1 * Range / 2
    weekly_pivot = (prev_high_w + prev_low_w + prev_close_w) / 3
    
    # Align daily levels to 4h timeframe
    daily_r4_array = np.full(len(df_1d), daily_r4)
    daily_s4_array = np.full(len(df_1d), daily_s4)
    daily_pivot_array = np.full(len(df_1d), daily_pivot)
    
    daily_r4_aligned = align_htf_to_ltf(prices, df_1d, daily_r4_array)
    daily_s4_aligned = align_htf_to_ltf(prices, df_1d, daily_s4_array)
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot_array)
    
    # Align weekly levels to 4h timeframe
    weekly_r4_array = np.full(len(df_1w), weekly_r4)
    weekly_s4_array = np.full(len(df_1w), weekly_s4)
    weekly_pivot_array = np.full(len(df_1w), weekly_pivot)
    
    weekly_r4_aligned = align_htf_to_ltf(prices, df_1w, weekly_r4_array)
    weekly_s4_aligned = align_htf_to_ltf(prices, df_1w, weekly_s4_array)
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_array)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values  # default to 1.0 if no MA
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(daily_r4_aligned[i]) or np.isnan(daily_s4_aligned[i]) or
            np.isnan(weekly_r4_aligned[i]) or np.isnan(weekly_s4_aligned[i]) or
            np.isnan(daily_pivot_aligned[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume filter (daily OR weekly)
        long_breakout = ((close[i] > daily_r4_aligned[i] or close[i] > weekly_r4_aligned[i]) and 
                         vol_ratio[i] > 1.5)
        short_breakout = ((close[i] < daily_s4_aligned[i] or close[i] < weekly_s4_aligned[i]) and 
                          vol_ratio[i] > 1.5)
        
        # Exit conditions: return to daily OR weekly pivot
        long_exit = close[i] < daily_pivot_aligned[i] or close[i] < weekly_pivot_aligned[i]
        short_exit = close[i] > daily_pivot_aligned[i] or close[i] > weekly_pivot_aligned[i]
        
        # Signal logic
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals