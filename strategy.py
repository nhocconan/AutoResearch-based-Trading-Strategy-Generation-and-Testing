#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h KAMA Trend with 4h/1d Confluence and Volume Filter
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# Combined with 4h trend (EMA21) and 1d trend (EMA50) for directional bias.
# Volume filter ensures institutional participation. Works in bull/bear via trend alignment.
# Target: 15-35 trades/year (60-140 over 4 years).

name = "1h_kama_trend_4h1d_confluence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate KAMA (adaptive moving average)
    close_s = pd.Series(close)
    change = abs(close_s.diff(10))
    volatility = abs(close_s.diff(1)).rolling(10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 4h EMA21 for trend
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=21, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Calculate 1d EMA50 for trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below KAMA or trend turns bearish
            if (close[i] <= kama[i] or 
                ema_4h_aligned[i] < ema_4h_aligned[i-1] or 
                ema_1d_aligned[i] < ema_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long
        elif position == -1:  # Short position
            # Exit: price closes above KAMA or trend turns bullish
            if (close[i] >= kama[i] or 
                ema_4h_aligned[i] > ema_4h_aligned[i-1] or 
                ema_1d_aligned[i] > ema_1d_aligned[i-1]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short
        else:  # Flat, look for entry
            # Long: price above KAMA and both trends bullish with volume
            if (close[i] > kama[i] and 
                ema_4h_aligned[i] > ema_4h_aligned[i-1] and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.20
            # Short: price below KAMA and both trends bearish with volume
            elif (close[i] < kama[i] and 
                  ema_4h_aligned[i] < ema_4h_aligned[i-1] and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.20
    
    return signals