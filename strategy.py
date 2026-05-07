#!/usr/bin/env python3
name = "6h_KAMA_Adaptive_Trend_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on 6h data
    def kama(close_series, period=10, fast_ema=2, slow_ema=30):
        change = np.abs(np.diff(close_series, n=period))
        volatility = np.sum(np.abs(np.diff(close_series)), axis=1)
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
        kama = np.full_like(close_series, np.nan, dtype=float)
        kama[period] = close_series[period]
        for i in range(period+1, len(close_series)):
            kama[i] = kama[i-1] + sc[i] * (close_series[i] - kama[i-1])
        return kama
    
    kama_values = kama(close, period=10, fast_ema=2, slow_ema=30)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Daily EMA34 for trend
    daily_ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema34_aligned = align_htf_to_ltf(prices, df_1d, daily_ema34)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 4  # ~1 day (4*6h) to prevent overtrading
    
    start_idx = max(30, 20)  # KAMA needs 30, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_values[i]) or 
            np.isnan(daily_ema34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close[i] > daily_ema34_aligned[i]
        trend_down = close[i] < daily_ema34_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price above KAMA with bullish daily trend and volume
            if (close[i] > kama_values[i] and 
                trend_up and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price below KAMA with bearish daily trend and volume
            elif (close[i] < kama_values[i] and 
                  trend_down and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below KAMA or daily trend changes to down
            if close[i] < kama_values[i] or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price rises above KAMA or daily trend changes to up
            if close[i] > kama_values[i] or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA adapts to market conditions - fast in trends, slow in ranges. Combined with daily EMA34 trend filter and volume confirmation, it captures strong trends while avoiding whipsaws in ranges. Works in bull markets (buy above KAMA in uptrend) and bear markets (sell below KAMA in downtrend). Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag. KAMA's adaptive nature reduces false signals during consolidation periods.