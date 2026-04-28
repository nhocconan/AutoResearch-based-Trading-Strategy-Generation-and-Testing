#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly EMA filter and volume confirmation
# KAMA adapts to market noise, reducing whipsaw in sideways markets while capturing trends.
# Weekly EMA(34) ensures alignment with the dominant trend, avoiding counter-trend trades.
# Volume confirmation (>1.8x 20-bar average) adds conviction to breakouts.
# Designed for 1d timeframe to target 7-25 trades/year, balancing frequency and signal quality.
# Works in both bull and bear markets by following the weekly trend filter.

name = "1d_KAMA_Trend_WeeklyEMA34_Volume_v1"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align weekly EMA to 1d (changes only when weekly bar closes)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate KAMA(10, 2, 30) on daily close
    # Efficiency Ratio (ER) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.subtract(close[10:], close[:-10]))  # |close[t] - close[t-10]|
    volatility = np.abs(np.subtract(close[1:], close[:-1]))  # |close[t] - close[t-1]|
    
    # Pad arrays for rolling sum
    change_padded = np.concatenate([np.full(10, np.nan), change])
    volatility_padded = np.concatenate([np.full(1, np.nan), volatility])
    
    # Rolling sum of volatility over 10 periods
    vol_sum = pd.Series(volatility_padded).rolling(window=10, min_periods=10).sum().values[10:]
    er = np.full_like(close, np.nan)
    er[10:] = np.divide(change, vol_sum, out=np.full_like(change, np.nan), where=vol_sum!=0)
    
    # Smoothing constants: fast = 2/(2+1) = 0.6667, slow = 2/(30+1) = 0.0645
    sc = (er * 0.6667 + (1 - er) * 0.0645) ** 2
    
    # Initialize KAMA
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # Start KAMA at index 9 (after 10 bars)
    
    # Calculate KAMA iteratively
    for i in range(10, n):
        if np.isnan(sc[i]) or np.isnan(kama[i-1]):
            kama[i] = np.nan
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 10, 20)  # Weekly EMA(34), KAMA(10,2,30), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > KAMA, above weekly EMA34, volume confirm
            if price > kama[i] and price > ema_34_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: Price < KAMA, below weekly EMA34, volume confirm
            elif price < kama[i] and price < ema_34_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement below KAMA or weekly EMA34
            if price < kama[i] or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on retracement above KAMA or weekly EMA34
            if price > kama[i] or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals