#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Load daily data for ATR calculation
    df_daily = get_htf_data(prices, '1d')
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate daily ATR(14)
    high_low = high_daily - low_daily
    high_close = np.abs(high_daily - np.roll(close_daily, 1))
    low_close = np.abs(low_daily - np.roll(close_daily, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr14_daily = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Align ATR to 12h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_daily, atr14_daily)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if np.isnan(ema50_aligned[i]) or np.isnan(atr_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr = atr_aligned[i]
        ema50 = ema50_aligned[i]
        
        if position == 0:
            # Long: price above weekly EMA50 and breaks above highest high of last 6 periods
            if price > ema50:
                highest_high = np.max(high[max(0, i-6):i])
                if price > highest_high:
                    signals[i] = 0.25
                    position = 1
            # Short: price below weekly EMA50 and breaks below lowest low of last 6 periods
            elif price < ema50:
                lowest_low = np.min(low[max(0, i-6):i])
                if price < lowest_low:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit: trailing stop based on ATR
            if position == 1:
                trailing_stop = np.max(high[max(0, i-12):i+1]) - 2.0 * atr
                if price < trailing_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                trailing_stop = np.min(low[max(0, i-12):i+1]) + 2.0 * atr
                if price > trailing_stop:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_EMA50_Trend_Breakout_ATR_Trailing_Stop"
timeframe = "12h"
leverage = 1.0