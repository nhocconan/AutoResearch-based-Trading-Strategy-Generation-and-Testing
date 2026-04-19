#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h RSI with 4h/1d trend filters and volume confirmation
# Uses RSI(14) overbought/oversold levels for mean reversion entries
# Filters: price must be above/below 4h EMA20 and 1d EMA50 for trend alignment
# Volume confirmation: current volume > 1.3x 20-period average
# Works in bull markets via buying dips in uptrend, in bear via selling rallies in downtrend
# Target: 15-35 trades/year to avoid fee drag
name = "1h_RSI_MeanReversion_TrendFilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for multi-timeframe analysis (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1d EMA50 for stronger trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]):
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
            # Long: RSI oversold (<30) + price above 4h EMA20 + price above 1d EMA50 + volume
            if rsi[i] < 30 and price > ema20_4h_aligned[i] and price > ema50_1d_aligned[i] and volume_filter:
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) + price below 4h EMA20 + price below 1d EMA50 + volume
            elif rsi[i] > 70 and price < ema20_4h_aligned[i] and price < ema50_1d_aligned[i] and volume_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought (>70) or price crosses below 4h EMA20
            if rsi[i] > 70 or price < ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI oversold (<30) or price crosses above 4h EMA20
            if rsi[i] < 30 or price > ema20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals