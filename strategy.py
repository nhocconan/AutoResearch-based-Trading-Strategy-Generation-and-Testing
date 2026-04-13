#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter + volume spike (>1.8x 24-period avg)
    # Long: price > H3 + price > 1d EMA34 + volume > 1.8x 24-period average volume
    # Short: price < L3 + price < 1d EMA34 + volume > 1.8x 24-period average volume
    # Exit: price crosses 1d EMA34 OR opposite Camarilla level touched (H4/L4)
    # Tight volume filter (1.8x) reduces trades to ~15-25/year for low fee drag
    # 1d EMA34 provides smooth trend filter, reducing whipsaw in ranging markets
    # Camarilla levels from 1d provide institutional support/resistance for breakouts
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 with min_periods
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_1d[33] = np.mean(close_1d[:34])  # SMA34 as seed
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Calculate 1d Camarilla levels (H3, L3, H4, L4) from previous day
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        
        if rang > 0:  # Avoid division by zero
            camarilla_h3[i] = prev_close + rang * 1.1 / 4
            camarilla_l3[i] = prev_close - rang * 1.1 / 4
            camarilla_h4[i] = prev_close + rang * 1.1 / 2
            camarilla_l4[i] = prev_close - rang * 1.1 / 2
        else:
            camarilla_h3[i] = camarilla_l3[i] = camarilla_h4[i] = camarilla_l4[i] = prev_close
    
    # Get 12h volume for confirmation (>1.8x 24-period average) - tighter filter
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    volume_spike = volume > (1.8 * vol_ma)
    
    # Align 1d indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla H3/L3
        long_breakout = close[i] > camarilla_h3_aligned[i]
        short_breakout = close[i] < camarilla_l3_aligned[i]
        
        # Trend filter from 1d EMA34
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: trend reversal OR price touches H4/L4 (stronger levels)
        long_exit = (close[i] < ema_1d_aligned[i]) or (close[i] >= camarilla_h4_aligned[i])
        short_exit = (close[i] > ema_1d_aligned[i]) or (close[i] <= camarilla_l4_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_h3l3_ema34_volume_v1"
timeframe = "12h"
leverage = 1.0