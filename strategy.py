#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 1d trend filter (EMA34) and volume confirmation
    # Uses 1d EMA34 for trend direction (HTF) to avoid counter-trend trades
    # Camarilla R4/S4 breakout on 6h for entry (strong continuation signals)
    # Volume > 1.5x 20-period average confirms breakout strength
    # Target: 12-30 trades/year (50-120 total over 4 years) for low fee drag
    # Works in bull via long bias, in bear via short bias from 1d EMA34 filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_1d[33] = np.mean(close_1d[:34])  # SMA34 as seed
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Get 6h Camarilla pivots (based on previous day's OHLC)
    # For 6h bars, we use the prior 1d bar's OHLC to calculate Camarilla levels
    camarilla_r4 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # Calculate pivots once per day using prior 1d bar
    for i in range(len(df_1d)):
        if i == 0:
            continue  # Need prior day
        prev_high = df_1d['high'].iloc[i-1]
        prev_low = df_1d['low'].iloc[i-1]
        prev_close = df_1d['close'].iloc[i-1]
        range_val = prev_high - prev_low
        
        # Camarilla levels
        r4 = prev_close + range_val * 1.1 / 2
        s4 = prev_close - range_val * 1.1 / 2
        
        # Find 6h bars that belong to this 1d period
        # Each 1d = 4 * 6h bars
        start_idx = i * 4
        end_idx = min((i + 1) * 4, n)
        for j in range(start_idx, end_idx):
            camarilla_r4[j] = r4
            camarilla_s4[j] = s4
    
    # Get 6h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align 1d EMA34 to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_r4[i]) or 
            np.isnan(camarilla_s4[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions at Camarilla R4/S4
        long_breakout = close[i] > camarilla_r4[i]
        short_breakout = close[i] < camarilla_s4[i]
        
        # Trend filter from 1d EMA34
        bullish_trend = close[i] > ema_1d_aligned[i]
        bearish_trend = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Breakout + trend alignment + volume confirmation
        long_entry = long_breakout and bullish_trend and volume_spike[i]
        short_entry = short_breakout and bearish_trend and volume_spike[i]
        
        # Exit logic: opposite breakout or trend reversal
        long_exit = short_breakout or (close[i] < ema_1d_aligned[i])
        short_exit = long_breakout or (close[i] > ema_1d_aligned[i])
        
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

name = "6h_1d_camarilla_breakout_ema34_volume_v1"
timeframe = "6h"
leverage = 1.0