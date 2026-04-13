#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# 12h_1w1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v2
# Hypothesis: 12h price breaks above/below weekly or daily Camarilla R4/S4 levels with volume > 2.0x 20-period average.
# Use the tighter (more restrictive) of weekly/daily levels to reduce false breakouts.
# Long when price breaks above min(weekly_R4, daily_R4) + volume condition.
# Short when price breaks below max(weekly_S4, daily_S4) + volume condition.
# Exit when price crosses weekly/daily pivot point (average of weekly PP and daily PP).
# Position size: 0.25 (25%).
# Designed for 12h timeframe to target 12-37 trades/year with volume confirmation reducing false signals.
# Expected to work in both bull (breakouts) and bear (fades from extremes) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous week's values for weekly calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    vol_1w = df_1w['volume'].values
    
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    prev_close_1w[0] = close_1w[0]
    
    # Previous day's values for daily calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Weekly VWAP approximation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    vwap_num_1w = np.cumsum(typical_price_1w * vol_1w)
    vwap_den_1w = np.cumsum(vol_1w)
    vwap_1w = np.where(vwap_den_1w != 0, vwap_num_1w / vwap_den_1w, typical_price_1w)
    
    # Daily VWAP approximation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    vwap_num_1d = np.cumsum(typical_price_1d * vol_1d)
    vwap_den_1d = np.cumsum(vol_1d)
    vwap_1d = np.where(vwap_den_1d != 0, vwap_num_1d / vwap_den_1d, typical_price_1d)
    
    # Weekly Camarilla calculation
    range_1w = prev_high_1w - prev_low_1w
    camarilla_pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    camarilla_r4_1w = camarilla_pp_1w + (range_1w * 1.1 / 2)
    camarilla_s4_1w = camarilla_pp_1w - (range_1w * 1.1 / 2)
    
    # Daily Camarilla calculation
    range_1d = prev_high_1d - prev_low_1d
    camarilla_pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    camarilla_r4_1d = camarilla_pp_1d + (range_1d * 1.1 / 2)
    camarilla_s4_1d = camarilla_pp_1d - (range_1d * 1.1 / 2)
    
    # Align weekly data to 12h
    camarilla_pp_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp_1w)
    camarilla_r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4_1w)
    camarilla_s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4_1w)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    vol_ma_20_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean()
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w.values)
    
    # Align daily data to 12h
    camarilla_pp_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp_1d)
    camarilla_r4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_1d)
    camarilla_s4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_1d)
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d.values)
    
    # Combined levels: use tighter (more restrictive) levels
    # For resistance: use the lower of weekly/daily R4 (harder to break)
    camarilla_r4_combined = np.minimum(camarilla_r4_1w_aligned, camarilla_r4_1d_aligned)
    # For support: use the higher of weekly/daily S4 (harder to break)
    camarilla_s4_combined = np.maximum(camarilla_s4_1w_aligned, camarilla_s4_1d_aligned)
    # Pivot: average of weekly and daily PP
    camarilla_pp_combined = (camarilla_pp_1w_aligned + camarilla_pp_1d_aligned) / 2.0
    
    # Volume condition: use the stronger of weekly/daily volume confirmation
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    vol_condition_1w = vol_1w_aligned > (vol_ma_20_1w_aligned * 2.0)
    vol_condition_1d = vol_1d_aligned > (vol_ma_20_1d_aligned * 2.0)
    vol_condition = vol_condition_1w | vol_condition_1d  # Either timeframe confirms
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_combined[i]) or np.isnan(camarilla_r4_combined[i]) or
            np.isnan(camarilla_s4_combined[i]) or np.isnan(vwap_1w_aligned[i]) or
            np.isnan(vwap_1d_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_r4_combined[i]
        short_breakout = close[i] < camarilla_s4_combined[i]
        
        # Exit condition
        long_exit = close[i] < camarilla_pp_combined[i]
        short_exit = close[i] > camarilla_pp_combined[i]
        
        if position == 0:
            if long_breakout and vol_condition[i]:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w1d_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v2"
timeframe = "12h"
leverage = 1.0