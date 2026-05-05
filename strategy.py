#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND weekly close > weekly EMA50 AND volume > 1.5x 20-day average
# Short when price breaks below 20-day Donchian low AND weekly close < weekly EMA50 AND volume > 1.5x 20-day average
# Exit when price crosses 10-day EMA (mean reversion to short-term trend)
# Uses 1d primary timeframe for lower trade frequency and better cost efficiency
# 1w HTF for trend filter to avoid counter-trend trades in bear markets
# Discrete sizing (0.30) to limit fee drag and manage drawdown
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume spike filter on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    if len(high) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Calculate 1d EMA10 for exit signal
    if len(close) >= 10:
        ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    else:
        ema_10 = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_10[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND weekly EMA50 uptrend AND volume spike
            if (close[i] > donchian_high[i] and 
                close_1w[i // (24*7)] > ema_50_1w[i // (24*7)] and  # Weekly alignment check
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian low AND weekly EMA50 downtrend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close_1w[i // (24*7)] < ema_50_1w[i // (24*7)] and  # Weekly alignment check
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below 10-day EMA (mean reversion)
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above 10-day EMA (mean reversion)
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals