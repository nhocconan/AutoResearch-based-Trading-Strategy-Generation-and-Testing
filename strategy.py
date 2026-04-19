#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with weekly trend filter and volume confirmation
# Uses KAMA to filter noise and identify low-noise trends
# Weekly trend filter ensures alignment with higher timeframe momentum
# Volume confirmation reduces false breakouts
# Designed to work in bull markets via trend following and in bear via trend reversals
# Target: 20-50 trades/year to minimize fee drag
name = "1d_KAMA_Trend_Filter_Weekly_Volume_v1"
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
    
    # Get weekly data for trend filter (ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend filter
    close_weekly = df_weekly['close'].values
    ema34_weekly = pd.Series(close_weekly).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle first er_period elements
    change = np.concatenate([np.full(er_period, np.nan), change])
    volatility = np.concatenate([np.full(er_period, np.nan), volatility[er_period-1:]])
    er = np.where(volatility != 0, change / volatility, 0)
    
    # Calculate Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]  # Start with first close after ER period
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # 10-period ATR for volatility filtering
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, er_period + 10)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema34_weekly_aligned[i]) or np.isnan(kama[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volume filter: current volume > 1.3x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.3 * avg_volume
        
        if position == 0:
            # Long: price crosses above KAMA + volume + weekly uptrend
            if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume_filter and price > ema34_weekly_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below KAMA + volume + weekly downtrend
            elif close[i] < kama[i] and close[i-1] >= kama[i-1] and volume_filter and price < ema34_weekly_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA or volatility filter fails
            if close[i] < kama[i] or volume[i] < 0.7 * np.mean(volume[max(0, i-20):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA or volatility filter fails
            if close[i] > kama[i] or volume[i] < 0.7 * np.mean(volume[max(0, i-20):i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals