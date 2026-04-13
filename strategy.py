# 4h_1d_camarilla_pivot_volume_breakout_v1
# Hypothesis: Camarilla pivot levels from daily timeframe act as strong support/resistance.
# Price breaking above/below these levels with volume confirmation and trend filter (EMA50) 
# provides high-probability entries. Works in both bull and bear markets by trading breakouts
# in the direction of the daily trend. Designed for low trade frequency to avoid fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla levels
    camarilla_high = np.full(len(close_1d), np.nan)
    camarilla_low = np.full(len(close_1d), np.nan)
    camarilla_close = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Camarilla levels: H4/L4 are most significant for breakouts
        camarilla_high[i] = pc + 1.1 * (ph - pl) / 2  # H4 level
        camarilla_low[i] = pc - 1.1 * (ph - pl) / 2   # L4 level
        camarilla_close[i] = pc
    
    # Calculate 50-period EMA on daily for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate volume moving average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_close_aligned = align_htf_to_ltf(prices, df_1d, camarilla_close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital - conservative sizing
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to daily EMA50
        above_ema = close[i] > ema_50_aligned[i]
        below_ema = close[i] < ema_50_aligned[i]
        
        # Volume confirmation: current volume above average
        volume_confirm = volume[i] > volume_ma[i]
        
        # Breakout conditions: price breaking Camarilla H4/L4 levels
        breakout_high = close[i] > camarilla_high_aligned[i]
        breakout_low = close[i] < camarilla_low_aligned[i]
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        long_entry = breakout_high and above_ema and volume_confirm
        short_entry = breakout_low and below_ema and volume_confirm
        
        # Exit conditions: price returns to Camarilla close level or trend reversal
        exit_long = position == 1 and (close[i] < camarilla_close_aligned[i] or below_ema)
        exit_short = position == -1 and (close[i] > camarilla_close_aligned[i] or above_ema)
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "4h_1d_camarilla_pivot_volume_breakout_v1"
timeframe = "4h"
leverage = 1.0