# 12h_GoldenCross_Trend_Filtered_Breakout
# Hypothesis: 12h timeframe reduces overtrading while capturing major trends.
# Uses weekly 200 EMA for long-term trend filter and daily 50/200 EMA cross for medium-term trend confirmation.
# Entry: Price breaks above 20-period high with volume confirmation in bullish trend alignment.
# Exit: Trailing stop based on ATR(14) from 4h timeframe.
# Designed for low trade frequency (<30/year) to minimize fee impact and work in both bull/bear markets.

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
    
    # Get weekly data for long-term trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Get daily data for medium-term trend (50/200 EMA cross)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Get 4h data for ATR calculation (used for stop loss)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(14) on 4h
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr_4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(atr_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        # Weekly trend: price above weekly 200 EMA
        weekly_uptrend = close[i] > ema_200_1w_aligned[i]
        weekly_downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Daily trend: 50 EMA above/below 200 EMA
        daily_uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        daily_downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Entry conditions: breakout of recent 20-period high/low with volume confirmation
        lookback = 20
        if i >= lookback:
            recent_high = np.nanmax(high[i-lookback:i])
            recent_low = np.nanmin(low[i-lookback:i])
            
            # Volume filter: current volume above 1.5x recent average
            vol_ma = np.nanmean(volume[max(0,i-10):i]) if i >= 10 else volume[i]
            volume_filter = volume[i] > 1.5 * vol_ma
            
            long_breakout = close[i] > recent_high
            short_breakout = close[i] < recent_low
            
            # Require both weekly and daily trend alignment for entry
            long_entry = weekly_uptrend and daily_uptrend and long_breakout and volume_filter
            short_entry = weekly_downtrend and daily_downtrend and short_breakout and volume_filter
        else:
            long_entry = False
            short_entry = False
        
        # Exit conditions: ATR-based trailing stop
        if position == 1:
            # Trail stop: exit if price drops 3*ATR from highest high since entry
            lookback_stop = min(25, i+1)
            recent_high = np.nanmax(high[i-lookback_stop:i+1])
            exit_condition = close[i] < recent_high - 3.0 * atr_4h_aligned[i]
        elif position == -1:
            # Trail stop: exit if price rises 3*ATR from lowest low since entry
            lookback_stop = min(25, i+1)
            recent_low = np.nanmin(low[i-lookback_stop:i+1])
            exit_condition = close[i] > recent_low + 3.0 * atr_4h_aligned[i]
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

name = "12h_GoldenCross_Trend_Filtered_Breakout"
timeframe = "12h"
leverage = 1.0