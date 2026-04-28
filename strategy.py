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
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for ATR calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(14) on 12h
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h = np.concatenate([[np.nan], tr_12h])  # Align with original index
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Bollinger Bands on 12h (20, 2)
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    upper_bb_12h = sma_20_12h + 2 * std_20_12h
    lower_bb_12h = sma_20_12h - 2 * std_20_12h
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb_12h)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions: Bollinger Band breakout with volume confirmation
        long_entry = uptrend and close[i] > upper_bb_aligned[i] and volume[i] > 1.5 * np.nanmedian(volume[max(0,i-10):i])
        short_entry = downtrend and close[i] < lower_bb_aligned[i] and volume[i] > 1.5 * np.nanmedian(volume[max(0,i-10):i])
        
        # Exit conditions: ATR-based stop loss
        if position == 1:
            # Trail stop: exit if price drops 2.5*ATR from highest high since entry
            recent_high = np.nanmax(high[max(0,i-20):i+1]) if i >= 20 else high[i]
            exit_condition = close[i] < recent_high - 2.5 * atr_12h_aligned[i]
        elif position == -1:
            # Trail stop: exit if price rises 2.5*ATR from lowest low since entry
            recent_low = np.nanmin(low[max(0,i-20):i+1]) if i >= 20 else low[i]
            exit_condition = close[i] > recent_low + 2.5 * atr_12h_aligned[i]
        else:
            exit_condition = False
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif exit_condition and position != 0:
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

name = "12h_BB_Breakout_Trend_ATR_1dEMA50"
timeframe = "12h"
leverage = 1.0