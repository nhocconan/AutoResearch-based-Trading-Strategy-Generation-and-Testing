# 1d_WeeklyTrend_SMA50_Trend_50_Signal
# Hypothesis: On daily timeframe, use weekly SMA50 trend filter with volume confirmation and low volatility filter.
# Long when price > weekly SMA50 AND volume > 1.5x 30-day avg AND daily ATR < its 50-day median (low vol).
# Short when price < weekly SMA50 with same filters.
# Exit when price crosses weekly SMA50.
# Weekly trend provides stability in both bull/bear markets; volume confirms conviction; low vol filter avoids choppy whipsaws.
# Target: 20-40 trades/year per symbol (<160 total over 4 years) to minimize fee drag.

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
    
    # Get weekly data for trend filter (HTF)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly SMA50 for trend filter
    close_weekly = pd.Series(df_weekly['close'].values)
    sma50_weekly = close_weekly.rolling(window=50, min_periods=50).mean().values
    sma50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, sma50_weekly)
    
    # Weekly ATR(14) for volatility
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_arr = df_weekly['close'].values
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - np.roll(close_weekly_arr, 1))
    tr3 = np.abs(low_weekly - np.roll(close_weekly_arr, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr14_weekly = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr14_weekly)
    
    # Volume filter: volume > 1.5x 30-day average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Volatility filter: ATR below its 50-day median (low volatility regime)
    atr_median = pd.Series(atr14_weekly_aligned).rolling(window=50, min_periods=14).median().values
    vol_filter = atr14_weekly_aligned < atr_median
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(sma50_weekly_aligned[i]) or np.isnan(atr14_weekly_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_median[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly SMA50 + volume filter + low volatility
            if (close[i] > sma50_weekly_aligned[i] and 
                volume_filter[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly SMA50 + volume filter + low volatility
            elif (close[i] < sma50_weekly_aligned[i] and 
                  volume_filter[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below weekly SMA50 (trend change)
            if close[i] < sma50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above weekly SMA50 (trend change)
            if close[i] > sma50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyTrend_SMA50_Trend_50_Signal"
timeframe = "1d"
leverage = 1.0