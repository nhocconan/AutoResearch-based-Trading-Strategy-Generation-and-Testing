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
    
    # Get 1d data for HTF calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    high_20 = np.full(len(close_1d), np.nan)
    low_20 = np.full(len(close_1d), np.nan)
    for i in range(20, len(close_1d)):
        high_20[i] = np.max(high_1d[i-20:i])
        low_20[i] = np.min(low_1d[i-20:i])
    
    # Calculate 30-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_30_1d = close_1d_series.ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Calculate 14-period ATR on 1d for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.append(close_1d[0], close_1d[:-1]))
    low_close = np.abs(low_1d - np.append(close_1d[0], close_1d[:-1]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr = pd.Series(tr)
    atr_14_1d = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align indicators to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_30_aligned = align_htf_to_ltf(prices, df_1d, ema_30_1d)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_30_aligned[i]) or
            np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA30
        above_ema = close[i] > ema_30_aligned[i]
        below_ema = close[i] < ema_30_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > high_20_aligned[i]
        short_breakout = close[i] < low_20_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated
        vol_filter = atr_14_aligned[i] > np.nanmedian(atr_14_aligned[max(0, i-50):i+1])
        
        # Entry conditions: breakout in direction of trend with volume confirmation
        long_entry = long_breakout and above_ema and vol_filter
        short_entry = short_breakout and below_ema and vol_filter
        
        # Exit conditions: opposite breakout or trend reversal
        exit_long = position == 1 and (short_breakout or below_ema)
        exit_short = position == -1 and (long_breakout or above_ema)
        
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

name = "4h_1d_donchian_ema30_vol_filter"
timeframe = "4h"
leverage = 1.0