#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Bollinger Band squeeze breakout + 1d trend filter + volume confirmation
    # Long when: BB squeeze (BW < 20th percentile) AND price breaks above upper BB AND 1d close > 1d EMA50 AND volume > 1.5x avg
    # Short when: BB squeeze AND price breaks below lower BB AND 1d close < 1d EMA50 AND volume > 1.5x avg
    # Exit when: price crosses BB middle OR volume drops below average
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1d EMA50 trend filter and BB squeeze capturing low-volatility breakouts.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2) on 6h
    lookback = 20
    ma = pd.Series(close).rolling(window=lookback, min_periods=lookback).mean().values
    std = pd.Series(close).rolling(window=lookback, min_periods=lookback).std().values
    upper_bb = ma + 2 * std
    lower_bb = ma - 2 * std
    bb_width = (upper_bb - lower_bb) / ma  # normalized bandwidth
    
    # Calculate 20th percentile of BB width for squeeze condition (using expanding window)
    bb_width_percentile = np.zeros(n)
    for i in range(lookback, n):
        bb_width_percentile[i] = np.percentile(bb_width[lookback:i+1], 20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(lookback, n):
        # Skip if data not ready
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_threshold[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Squeeze condition: BB width below 20th percentile
        squeeze = bb_width[i] < bb_width_percentile[i]
        
        # Breakout conditions
        long_breakout = close[i] > upper_bb[i]
        short_breakout = close[i] < lower_bb[i]
        
        # 1d trend filter
        long_trend = close[i] > ema_50_1d_aligned[i]
        short_trend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Entry conditions
        long_entry = squeeze and long_breakout and long_trend and vol_ok and position != 1
        short_entry = squeeze and short_breakout and short_trend and vol_ok and position != -1
        
        # Exit conditions: price crosses BB middle OR volume drops below average
        exit_long = close[i] < ma[i] or volume[i] < vol_ma[i]
        exit_short = close[i] > ma[i] or volume[i] < vol_ma[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
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

name = "6h_1d_bb_squeeze_trend_volume_v1"
timeframe = "6h"
leverage = 1.0