#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume spike + RSI filter + daily trend filter
# Long when price breaks above Donchian high(20) with volume > 1.5x average and RSI > 50 and daily EMA50 up
# Short when price breaks below Donchian low(20) with volume > 1.5x average and RSI < 50 and daily EMA50 down
# Uses volume confirmation to avoid false breakouts, RSI to avoid overextended moves, daily trend for alignment
# Targets 20-50 trades per year to minimize fee drag while capturing strong moves

name = "4h_Donchian20_Volume_RSI_DailyTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    daily_close = df_1d['close'].values
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Enter long: break above Donchian high + volume spike + RSI > 50 + daily uptrend
            if (close[i] > donch_high[i] and vol_ratio > 1.5 and 
                rsi[i] > 50 and ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low + volume spike + RSI < 50 + daily downtrend
            elif (close[i] < donch_low[i] and vol_ratio > 1.5 and 
                  rsi[i] < 50 and ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: break below Donchian low or loss of daily uptrend
            if close[i] < donch_low[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: break above Donchian high or loss of daily downtrend
            if close[i] > donch_high[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals