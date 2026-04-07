#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h KAMA + 4h/1d Trend + Volume Confirmation
# Hypothesis: KAMA adapts to market noise - in low volatility it tracks price closely (trend following),
# in high volatility it smooths out noise (mean reversion). We use 4h/1d for trend direction
# and 1h for precise entries. Volume confirms institutional participation. Works in bull/bear
# by adapting to volatility regime. Target: 15-37 trades/year (60-150 over 4 years).
name = "1h_kama_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # KAMA (Adaptive Moving Average) on 1h timeframe
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_1h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_filter = volume > (vol_ma * 1.5)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(kama[i]) or np.isnan(ema_4h_1h[i]) or 
            np.isnan(ema_1d_1h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below KAMA or trend breaks
            if close[i] < kama[i] or close[i] < ema_4h_1h[i] or close[i] < ema_1d_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above KAMA or trend breaks
            if close[i] > kama[i] or close[i] > ema_4h_1h[i] or close[i] > ema_1d_1h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume confirmation
            if vol_filter[i]:
                # Long: price above KAMA and both 4h/1d EMAs (strong uptrend)
                if close[i] > kama[i] and close[i] > ema_4h_1h[i] and close[i] > ema_1d_1h[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price below KAMA and both 4h/1d EMAs (strong downtrend)
                elif close[i] < kama[i] and close[i] < ema_4h_1h[i] and close[i] < ema_1d_1h[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals