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
    
    # Get daily data for HTF context (Camarilla levels and volatility)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate daily range for pivot calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    daily_range = high_1d - low_1d
    
    # Daily Camarilla pivot levels (based on previous day)
    camarilla_r3 = close_1d + daily_range * 1.1 / 4
    camarilla_s3 = close_1d - daily_range * 1.1 / 4
    
    # Align Daily Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Weekly trend filter: price above/below weekly SMA20
    close_1w_series = pd.Series(df_1w['close'].values)
    sma20_1w = close_1w_series.rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Volatility filter: daily ATR ratio (current vs 20-day average)
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - df_1d['close'].shift())
    low_close = np.abs(df_1d['low'] - df_1d['close'].shift())
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_1d / atr_ma_1d
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Hour filter: 0-24 UTC (trade all hours for 12h timeframe)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(sma20_1w_aligned[i]) or np.isnan(atr_ratio_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated (above average)
        vol_filter = atr_ratio_aligned[i] > 1.0
        
        # Trend filter: price above/below weekly SMA20
        trend_up = close[i] > sma20_1w_aligned[i]
        trend_down = close[i] < sma20_1w_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above daily R3 with volatility and trend up
        # Short: price breaks below daily S3 with volatility and trend down
        long_entry = (close[i] > r3_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < s3_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: price returns to opposite daily S3/R3 levels
        long_exit = (close[i] < s3_aligned[i])
        short_exit = (close[i] > r3_aligned[i])
        
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

name = "12h_DailyCamarilla_R3S3_WeeklyTrend_VolatilityFilter"
timeframe = "12h"
leverage = 1.0