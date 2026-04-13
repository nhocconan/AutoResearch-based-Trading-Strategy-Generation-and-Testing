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
    
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2) on 12h
    sma_20 = close_12h_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_12h_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Get 1d data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 20-period SMA on 1d
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    rsi_14_aligned = align_htf_to_ltf(prices, df_12h, rsi_14)
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_20_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i]) or
            np.isnan(sma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA20
        above_ema = close[i] > ema_20_aligned[i]
        below_ema = close[i] < ema_20_aligned[i]
        
        # RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Bollinger Band conditions: price near bands
        near_upper_band = close[i] > bb_upper_aligned[i] * 0.98
        near_lower_band = close[i] < bb_lower_aligned[i] * 1.02
        
        # Daily trend filter: price above/below daily SMA20
        above_daily_sma = close[i] > sma_20_1d_aligned[i]
        below_daily_sma = close[i] < sma_20_1d_aligned[i]
        
        # Entry conditions
        long_entry = above_ema and rsi_not_overbought and near_lower_band and above_daily_sma
        short_entry = below_ema and rsi_not_oversold and near_upper_band and below_daily_sma
        
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

name = "12h_ema_rsi_bb_daily_filter"
timeframe = "6h"
leverage = 1.0