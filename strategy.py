#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2) on 4h
    close_4h_series = pd.Series(close_4h)
    sma_20 = close_4h_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_4h_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Get 1d data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 50-period EMA on 1d
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to hourly timeframe
    rsi_14_aligned = align_htf_to_ltf(prices, df_4h, rsi_14)
    bb_upper_aligned = align_htf_to_ltf(prices, df_4h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_4h, bb_lower)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h Bollinger Bands for entry timing
    close_series = pd.Series(close)
    sma_20_1h = close_series.rolling(window=20, min_periods=20).mean().values
    std_20_1h = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper_1h = sma_20_1h + 2 * std_20_1h
    bb_lower_1h = sma_20_1h - 2 * std_20_1h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # 4h trend filter: price above/below EMA50 on 1d
        above_ema = close[i] > ema_50_1d_aligned[i]
        below_ema = close[i] < ema_50_1d_aligned[i]
        
        # 4h RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # 4h Bollinger Band conditions: price near bands
        near_upper_band = close[i] > bb_upper_aligned[i] * 0.98
        near_lower_band = close[i] < bb_lower_aligned[i] * 1.02
        
        # 1h Bollinger Band conditions: price near bands for entry timing
        near_upper_1h = close[i] > bb_upper_1h[i] * 0.98
        near_lower_1h = close[i] < bb_lower_1h[i] * 1.02
        
        # Entry conditions
        long_entry = above_ema and rsi_not_overbought and near_lower_band and near_lower_1h
        short_entry = below_ema and rsi_not_oversold and near_upper_band and near_upper_1h
        
        # Exit conditions: opposite signal or RSI extreme
        exit_long = position == 1 and (below_ema or rsi_14_aligned[i] > 75)
        exit_short = position == -1 and (above_ema or rsi_14_aligned[i] < 25)
        
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

name = "1h_4h_1d_rsi_bb_ema_filter"
timeframe = "1h"
leverage = 1.0