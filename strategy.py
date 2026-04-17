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
    
    # Get daily data for weekly SMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 200-day SMA on daily data
    close_1d_series = pd.Series(close_1d)
    sma200_1d = close_1d_series.rolling(window=200, min_periods=200).mean().values
    
    # Align daily 200SMA to 1d timeframe (current day's value)
    sma200_1d_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 50-week SMA on weekly data
    close_1w_series = pd.Series(close_1w)
    sma50_1w = close_1w_series.rolling(window=50, min_periods=50).mean().values
    
    # Align weekly 50SMA to 1d timeframe (previous week's value to avoid look-ahead)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    # Calculate 14-day ATR for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # Need sufficient data for SMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(sma200_1d_aligned[i]) or np.isnan(sma50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 200-day SMA
        price_above_sma200 = close[i] > sma200_1d_aligned[i]
        price_below_sma200 = close[i] < sma200_1d_aligned[i]
        
        # Weekly trend filter: weekly price above/below 50-week SMA
        weekly_uptrend = sma50_1w_aligned[i] > sma50_1w_aligned[i-1] if i > 0 else False
        weekly_downtrend = sma50_1w_aligned[i] < sma50_1w_aligned[i-1] if i > 0 else False
        
        # Volatility filter: avoid extremely low volatility periods
        vol_filter = atr[i] > 0.01 * close[i]  # ATR > 1% of price
        
        if position == 0:
            # Long: price above 200-day SMA + weekly uptrend + volatility filter
            if (price_above_sma200 and weekly_uptrend and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price below 200-day SMA + weekly downtrend + volatility filter
            elif (price_below_sma200 and weekly_downtrend and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below 200-day SMA or weekly trend turns down
            if (close[i] < sma200_1d_aligned[i] or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above 200-day SMA or weekly trend turns up
            if (close[i] > sma200_1d_aligned[i] or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_SMA200_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0