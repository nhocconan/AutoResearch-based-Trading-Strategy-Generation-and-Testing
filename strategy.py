#!/usr/bin/env python3
"""
1d_1w_RSI_Momentum_with_Volume_Filter
Hypothesis: Trade weekly RSI extremes on daily chart with volume confirmation.
Buy when weekly RSI < 30 and daily RSI crosses above 30 with volume > 1.5x 20-day average.
Sell when weekly RSI > 70 and daily RSI crosses below 70 with volume > 1.5x 20-day average.
Uses weekly trend filter: only long when price above weekly 50 EMA, short when below.
Designed for low trade frequency (~10-25/year) with high conviction in mean reversion during extremes.
Works in bull markets (buying dips in uptrend) and bear markets (selling rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_RSI_Momentum_with_Volume_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY DATA FOR RSI AND TREND ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI (14-period)
    def calculate_rsi(series, period=14):
        delta = np.diff(series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(series)
        avg_loss = np.zeros_like(series)
        
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        for i in range(period + 1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_1w = calculate_rsi(close_1w, 14)
    
    # Weekly 50 EMA for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === DAILY INDICATORS ===
    # Daily RSI for entry timing
    rsi_1d = calculate_rsi(close, 14)
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(rsi_1d[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine weekly trend
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume strength
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # RSI conditions
        rsi_overbought = rsi_1w_aligned[i] > 70
        rsi_oversold = rsi_1w_aligned[i] < 30
        
        # Daily RSI cross signals
        rsi_cross_up = (rsi_1d[i] > 30) and (rsi_1d[i-1] <= 30) if i > 0 else False
        rsi_cross_down = (rsi_1d[i] < 70) and (rsi_1d[i-1] >= 70) if i > 0 else False
        
        # Long: weekly oversold + daily RSI crosses up + volume + weekly uptrend
        long_signal = (rsi_oversold and 
                      rsi_cross_up and 
                      strong_volume and 
                      uptrend)
        
        # Short: weekly overbought + daily RSI crosses down + volume + weekly downtrend
        short_signal = (rsi_overbought and 
                       rsi_cross_down and 
                       strong_volume and 
                       downtrend)
        
        # Exit: opposite RSI extreme or weekly trend reversal
        exit_long = (position == 1 and 
                    (rsi_1w_aligned[i] > 70 or not uptrend))
        exit_short = (position == -1 and 
                     (rsi_1w_aligned[i] < 30 or not downtrend))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals