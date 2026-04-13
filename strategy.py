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
    
    # Get 4h data for calculations (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 4-period RSI on 4h for short-term momentum
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/4, adjust=False, min_periods=4).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    
    # Calculate 14-period RSI on 1d for longer-term momentum
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = avg_gain_1d / (avg_loss_1d + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    
    # Calculate Bollinger Bands (20, 2) on 1d
    close_1d_series = pd.Series(close_1d)
    sma_20_1d = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std_20_1d = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper_1d = sma_20_1d + 2 * std_20_1d
    bb_lower_1d = sma_20_1d - 2 * std_20_1d
    
    # Align indicators to 4h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper_1d)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # RSI conditions: look for divergence between short and long term
        rsi_4h_oversold = rsi_4h_aligned[i] < 30
        rsi_4h_overbought = rsi_4h_aligned[i] > 70
        rsi_1d_bullish = rsi_1d_aligned[i] > 50
        rsi_1d_bearish = rsi_1d_aligned[i] < 50
        
        # Bollinger Band conditions: price near bands
        near_upper_band = close[i] > bb_upper_aligned[i] * 0.98
        near_lower_band = close[i] < bb_lower_aligned[i] * 1.02
        
        # Entry conditions
        long_entry = rsi_4h_oversold and rsi_1d_bullish and near_lower_band
        short_entry = rsi_4h_overbought and rsi_1d_bearish and near_upper_band
        
        # Exit conditions: opposite RSI signal or middle band
        exit_long = position == 1 and (rsi_4h_aligned[i] > 50 or close[i] > sma_20_1d[i] if not np.isnan(sma_20_1d[i]) else False)
        exit_short = position == -1 and (rsi_4h_aligned[i] < 50 or close[i] < sma_20_1d[i] if not np.isnan(sma_20_1d[i]) else False)
        
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

name = "4h_1d_rsi_divergence_bb"
timeframe = "4h"
leverage = 1.0