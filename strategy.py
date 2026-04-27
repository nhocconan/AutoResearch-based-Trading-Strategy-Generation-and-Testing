#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume spike
# Donchian channels capture price structure; breakouts with volume and higher timeframe
# trend capture momentum while minimizing false breaks. Works in bull/bear by filtering
# breakout direction with 1d EMA trend. Target: 50-150 total trades over 4 years (~12-37/year).
# Uses discrete position sizing (0.25) to minimize churn and manage drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels for each 1d bar (20-period lookback)
    upper = np.full(len(df_1d), np.nan)
    lower = np.full(len(df_1d), np.nan)
    
    for i in range(20, len(df_1d)):
        upper[i] = np.max(high_1d[i-20:i])
        lower[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian channels to 12h timeframe (wait for 1d close)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2.0 x 12-period average (1.5 days of 12h bars)
    vol_ma_12 = np.full(n, np.nan)
    for i in range(11, n):
        vol_ma_12[i] = np.mean(volume[i-11:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d data (20 bars), EMA (50), volume MA (12)
    start_idx = max(20, 50, 11)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_12[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_12[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: break above upper band with volume and bullish trend
            if price > upper_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: break below lower band with volume and bearish trend
            elif price < lower_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower band (mean reversion) or trend turns bearish
            if price <= lower_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper band (mean reversion) or trend turns bullish
            if price >= upper_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian_20_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0