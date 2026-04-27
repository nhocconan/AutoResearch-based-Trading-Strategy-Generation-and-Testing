# US1DC5_MacroTrend_Refined
# Hypothesis: Daily MACD crossover with 1-week EMA filter and volume confirmation captures major trend shifts
# Works in bull/bear by requiring both momentum (MACD) and trend (weekly EMA) alignment
# Target: 20-40 trades/year to minimize fee drag while capturing sustained moves
# Uses daily timeframe with weekly trend filter for institutional-grade signals

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
    
    # Get weekly data for higher timeframe trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA(21) for trend filter
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily MACD (12,26,9)
    close_series = pd.Series(close)
    ema_12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close_series.ewm(span=26, adjust=False, min_periods=26).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_hist = macd_line - signal_line
    
    macd_line_vals = macd_line.values
    signal_line_vals = signal_line.values
    macd_hist_vals = macd_hist.values
    
    # Calculate daily volume moving average for confirmation
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(macd_line_vals[i]) or 
            np.isnan(signal_line_vals[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA21
        above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        
        # MACD conditions: bullish/bearish crossover
        macd_bullish = macd_line_vals[i] > signal_line_vals[i] and macd_line_vals[i-1] <= signal_line_vals[i-1]
        macd_bearish = macd_line_vals[i] < signal_line_vals[i] and macd_line_vals[i-1] >= signal_line_vals[i-1]
        
        # Volume confirmation: above average volume
        vol_confirm = volume[i] > volume_ma_20[i]
        
        # Long conditions: above weekly EMA + MACD bullish crossover + volume
        long_condition = above_weekly_ema and macd_bullish and vol_confirm
        
        # Short conditions: below weekly EMA + MACD bearish crossover + volume
        short_condition = below_weekly_ema and macd_bearish and vol_confirm
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite MACD crossover
        elif position == 1 and macd_bearish:
            signals[i] = 0.0
            position = 0
        elif position == -1 and macd_bullish:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "US1DC5_MacroTrend_Refined"
timeframe = "1d"
leverage = 1.0