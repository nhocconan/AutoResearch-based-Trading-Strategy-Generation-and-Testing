#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d trend filter
    # Long: Close breaks above H4 with volume > 1.2x average AND 12h EMA50 rising
    # Short: Close breaks below L4 with volume > 1.2x average AND 12h EMA50 falling
    # Exit: Price retracement to H3/L3 or volume dry-up
    # Using 6h timeframe for balance of signal quality and trade frequency,
    # Camarilla levels from prior day for structure, volume for confirmation,
    # 12h EMA for trend filter to avoid counter-trend trades.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (prior day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior daily OHLC
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.125 * (High - Low)
    # L3 = Close - 1.125 * (High - Low)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's data (no look-ahead)
    close_1d_lag = np.roll(close_1d, 1)
    high_1d_lag = np.roll(high_1d, 1)
    low_1d_lag = np.roll(low_1d, 1)
    close_1d_lag[0] = np.nan
    high_1d_lag[0] = np.nan
    low_1d_lag[0] = np.nan
    
    # Calculate Camarilla levels
    H4 = close_1d_lag + 1.5 * (high_1d_lag - low_1d_lag)
    L4 = close_1d_lag - 1.5 * (high_1d_lag - low_1d_lag)
    H3 = close_1d_lag + 1.125 * (high_1d_lag - low_1d_lag)
    L3 = close_1d_lag - 1.125 * (high_1d_lag - low_1d_lag)
    
    # Align daily Camarilla levels to 6h
    H4_6h = align_htf_to_ltf(prices, df_1d, H4)
    L4_6h = align_htf_to_ltf(prices, df_1d, L4)
    H3_6h = align_htf_to_ltf(prices, df_1d, H3)
    L3_6h = align_htf_to_ltf(prices, df_1d, L3)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA direction: 1 = rising, -1 = falling
    ema_dir_12h = np.zeros_like(ema_12h)
    ema_dir_12h[1:] = np.where(ema_12h[1:] > ema_12h[:-1], 1, 
                               np.where(ema_12h[1:] < ema_12h[:-1], -1, 0))
    # Forward fill to handle initial NaN
    for i in range(1, len(ema_dir_12h)):
        if ema_dir_12h[i] == 0:
            ema_dir_12h[i] = ema_dir_12h[i-1]
    # Align 12h EMA direction to 6h
    ema_dir_12h_6h = align_htf_to_ltf(prices, df_12h, ema_dir_12h)
    
    # Get 6h volume for confirmation (>1.2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.2 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H4_6h[i]) or np.isnan(L4_6h[i]) or 
            np.isnan(H3_6h[i]) or np.isnan(L3_6h[i]) or
            np.isnan(ema_dir_12h_6h[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Breakout conditions
        long_breakout = close[i] > H4_6h[i]
        short_breakout = close[i] < L4_6h[i]
        
        # Retracement exit conditions
        long_exit = close[i] < H3_6h[i]
        short_exit = close[i] > L3_6h[i]
        
        # Entry logic: breakout + trend filter + volume confirmation
        long_entry = long_breakout and (ema_dir_12h_6h[i] == 1) and vol_confirm
        short_entry = short_breakout and (ema_dir_12h_6h[i] == -1) and vol_confirm
        
        # Exit logic: retracement or volume dry-up
        long_exit_cond = long_exit or not vol_confirm
        short_exit_cond = short_exit or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit_cond:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit_cond:
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

name = "6h_12h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0