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
    
    # Get 12h data once for HTF context
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 12h high/low/close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h range for pivot calculations
    daily_range_12h = high_12h - low_12h
    
    # Calculate 1d high/low/close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d range for pivot calculations
    daily_range_1d = high_1d - low_1d
    
    # Calculate ATR (14) for volatility filter
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h ATR for volatility filter
    tr_12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # 12h ATR-based volatility filter: current 12h ATR > 1.5 * average 12h ATR
    atr_ma_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    vol_filter = atr_12h > (atr_ma_12h * 1.5)
    
    # Calculate 12h EMA21 for trend
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d EMA50 for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 4h timeframe
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_filter_aligned = align_htf_to_ltf(prices, df_12h, vol_filter)
    
    # Calculate 12h ATR-based volatility
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_filter_aligned[i]) or np.isnan(atr_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: 12h ATR > 1.5 * average 12h ATR
        vol_condition = vol_filter_aligned[i]
        
        # Trend filter: 12h EMA21 > 1d EMA50 (bullish alignment)
        bullish_alignment = ema_21_12h_aligned[i] > ema_50_1d_aligned[i]
        bearish_alignment = ema_21_12h_aligned[i] < ema_50_1d_aligned[i]
        
        # Entry conditions:
        # Long: bullish alignment + volatility expansion
        # Short: bearish alignment + volatility expansion
        long_entry = bullish_alignment and vol_condition
        short_entry = bearish_alignment and vol_condition
        
        # Exit conditions: trend reversal or volatility contraction
        long_exit = (not bullish_alignment) or (not vol_condition)
        short_exit = (not bearish_alignment) or (not vol_condition)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Volatility_Trend_Alignment_12h1d"
timeframe = "4h"
leverage = 1.0