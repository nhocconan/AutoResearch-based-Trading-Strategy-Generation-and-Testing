# 1d_1w_sma_trend_v1
# 1d trend filter with 1w SMA confirmation and volume filter
# Hypothesis: Trend following on daily with weekly trend filter reduces whipsaw in bear markets
# Works in bull by riding trends, works in bear by avoiding counter-trend trades via weekly filter
# Target: 20-50 trades/year, low turnover to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_sma_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for SMA calculation (already daily, but use for consistency)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma_50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly SMA to daily timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    # Volume filter - 20-period average on daily data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(sma_50_1d[i]) or np.isnan(sma_50_1w_aligned[i]) or 
            np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend conditions
        price_above_daily_sma = close[i] > sma_50_1d[i]
        price_below_daily_sma = close[i] < sma_50_1d[i]
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        # Entry signals with volume confirmation
        # Long: price above daily SMA AND weekly uptrend AND volume
        long_signal = price_above_daily_sma and weekly_uptrend and volume_ok[i]
        # Short: price below daily SMA AND weekly downtrend AND volume
        short_signal = price_below_daily_sma and weekly_downtrend and volume_ok[i]
        
        # Exit when trend changes
        exit_long = price_below_daily_sma or not weekly_uptrend
        exit_short = price_above_daily_sma or not weekly_downtrend
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals