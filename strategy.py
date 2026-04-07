#!/usr/bin/env python3
"""
1d_1w_trend_with_volume_and_volatility_filter
Hypothesis: On daily timeframe, use weekly EMA trend filter with daily EMA crossover for entry,
confirmed by volume and low volatility (ATR-based). Weekly trend ensures we trade with the
higher-timeframe momentum, while daily EMA captures medium-term swings. Volume confirms
institutional participation, and low volatility filter avoids choppy markets. Designed for
20-50 total trades over 4 years (~5-12/year) to minimize fee decay and perform in both
bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trend_with_volume_and_volatility_filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA periods
    daily_ema_fast = 10
    daily_ema_slow = 30
    weekly_ema_trend = 20
    
    # Calculate daily EMAs
    close_series = pd.Series(close)
    daily_ema_fast_values = close_series.ewm(span=daily_ema_fast, adjust=False, min_periods=daily_ema_fast).mean().values
    daily_ema_slow_values = close_series.ewm(span=daily_ema_slow, adjust=False, min_periods=daily_ema_slow).mean().values
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    weekly_close = df_weekly['close'].values
    weekly_ema_values = pd.Series(weekly_close).ewm(span=weekly_ema_trend, adjust=False, min_periods=weekly_ema_trend).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema_values)
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Volatility filter: ATR(14) < 50-day ATR average
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(daily_ema_slow, 20, 50)
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(daily_ema_fast_values[i]) or np.isnan(daily_ema_slow_values[i]) or 
            np.isnan(weekly_ema_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or np.isnan(atr_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume and volatility filters
        vol_ok = volume[i] > vol_ma[i]
        vol_filter_ok = atr[i] < atr_ma[i]  # Low volatility regime
        
        if position == 1:  # Long position
            # Exit: daily bearish crossover OR weekly trend turns bearish
            if daily_ema_fast_values[i] <= daily_ema_slow_values[i] or close[i] < weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: daily bullish crossover OR weekly trend turns bullish
            if daily_ema_fast_values[i] >= daily_ema_slow_values[i] or close[i] > weekly_ema_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and vol_filter_ok:
                # Bullish crossover: fast EMA crosses above slow EMA AND price above weekly EMA
                if (daily_ema_fast_values[i] > daily_ema_slow_values[i] and 
                    daily_ema_fast_values[i-1] <= daily_ema_slow_values[i-1] and
                    close[i] > weekly_ema_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Bearish crossover: fast EMA crosses below slow EMA AND price below weekly EMA
                elif (daily_ema_fast_values[i] < daily_ema_slow_values[i] and 
                      daily_ema_fast_values[i-1] >= daily_ema_slow_values[i-1] and
                      close[i] < weekly_ema_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals