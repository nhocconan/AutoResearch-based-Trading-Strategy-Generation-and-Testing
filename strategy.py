#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for calculations
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 20-period EMA on 12h (trend filter)
    close_12h_series = pd.Series(close_12h)
    ema_20_12h = close_12h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channel (20) on 12h
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14) on 12h for volatility filter
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to primary timeframe (4h)
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    highest_high_aligned = align_htf_to_ltf(prices, df_12h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_12h, lowest_low)
    atr_14_aligned = align_htf_to_ltf(prices, df_12h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(highest_high_aligned[i]) or
            np.isnan(lowest_low_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA20
        above_ema = close[i] > ema_20_aligned[i]
        below_ema = close[i] < ema_20_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_aligned[i]
        breakout_down = close[i] < lowest_low_aligned[i]
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])
        
        # Entry conditions
        long_entry = breakout_up and above_ema and vol_filter
        short_entry = breakout_down and below_ema and vol_filter
        
        # Exit conditions: opposite breakout or volatility collapse
        exit_long = position == 1 and (breakout_down or atr_14_aligned[i] < 0.5 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1]))
        exit_short = position == -1 and (breakout_up or atr_14_aligned[i] < 0.5 * np.nanmedian(atr_14_aligned[max(0, i-50):i+1]))
        
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

name = "4h_12h_donchian_ema_vol_filter"
timeframe = "4h"
leverage = 1.0