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
    
    # Get 1d data for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 10-period EMA on 1d (trend filter)
    close_1d_series = pd.Series(close_1d)
    ema_10_1d = close_1d_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands (20, 2) on 1d
    sma_20 = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    
    # Align indicators to 4h timeframe
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.30  # 30% position size
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_10_aligned[i]) or 
            np.isnan(rsi_14_aligned[i]) or
            np.isnan(bb_upper_aligned[i]) or
            np.isnan(bb_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below EMA10
        above_ema = close[i] > ema_10_aligned[i]
        below_ema = close[i] < ema_10_aligned[i]
        
        # RSI conditions: not overbought/oversold
        rsi_not_overbought = rsi_14_aligned[i] < 70
        rsi_not_oversold = rsi_14_aligned[i] > 30
        
        # Bollinger Band conditions: price near bands
        near_upper_band = close[i] > bb_upper_aligned[i] * 0.98
        near_lower_band = close[i] < bb_lower_aligned[i] * 1.02
        
        # Entry conditions
        long_entry = above_ema and rsi_not_overbought and near_lower_band
        short_entry = below_ema and rsi_not_oversold and near_upper_band
        
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

name = "4h_1d_ema_rsi_bb"
timeframe = "4h"
leverage = 1.0