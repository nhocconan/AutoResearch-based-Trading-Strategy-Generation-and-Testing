#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h primary with 1d HTF - Camarilla pivot breakout with volume confirmation
    # Designed to capture institutional breakouts at key daily levels with volume confirmation
    # Target: 75-200 trades over 4 years (19-50/year) for low fee drag and good generalization
    # Works in both bull and bear markets by trading breakouts in direction of 1d trend
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values if 'volume' in df_4h.columns else np.ones(len(df_4h))
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla: H4 = Close + 1.5*(High-Low), L4 = Close - 1.5*(High-Low)
    # We use the previous day's range to calculate today's levels
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Camarilla levels (H4/L4 are the key breakout levels)
    camarilla_h4 = prev_close_1d + 1.5 * (prev_high_1d - prev_low_1d)
    camarilla_l4 = prev_close_1d - 1.5 * (prev_high_1d - prev_low_1d)
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 4h primary timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8x 20-period average
        volume_confirmed = volume_4h[i] > 1.8 * vol_avg_20_aligned[i]
        
        # Breakout conditions at Camarilla H4/L4 levels
        breakout_up = close_4h[i] > camarilla_h4_aligned[i] if 'close_4h' in locals() else close[i] > camarilla_h4_aligned[i]
        breakout_down = close_4h[i] < camarilla_l4_aligned[i] if 'close_4h' in locals() else close[i] < camarilla_l4_aligned[i]
        
        # Trend filter: only trade in direction of 1d EMA200
        # For long: price above EMA200; for short: price below EMA200
        trend_filter_long = close[i] > ema200_1d_aligned[i]
        trend_filter_short = close[i] < ema200_1d_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and volume_confirmed and trend_filter_long
        enter_short = breakout_down and volume_confirmed and trend_filter_short
        
        # Exit conditions: price returns to previous day's close (pivot point)
        exit_long = position == 1 and close[i] <= prev_close_1d_aligned if 'prev_close_1d_aligned' in locals() else close[i] <= np.roll(close_1d, 1)[i] if i < len(np.roll(close_1d, 1)) else False
        exit_short = position == -1 and close[i] >= prev_close_1d_aligned if 'prev_close_1d_aligned' in locals() else close[i] >= np.roll(close_1d, 1)[i] if i < len(np.roll(close_1d, 1)) else False
        
        # Simplify exit: exit when price crosses the Camarilla H3/L3 levels (closer to mean)
        camarilla_h3 = prev_close_1d + 1.125 * (prev_high_1d - prev_low_1d)
        camarilla_l3 = prev_close_1d - 1.125 * (prev_high_1d - prev_low_1d)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        
        exit_long = position == 1 and close[i] <= camarilla_h3_aligned[i]
        exit_short = position == -1 and close[i] >= camarilla_l3_aligned[i]
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0