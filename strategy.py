#!/usr/bin/env python3
name = "1h_KAMA_Trend_Filter_VolumeSpike"
timeframe = "1h"
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
    
    # Daily trend filter (1d EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema1d_aligned = align_htf_to_ltf(prices, df_1d, ema1d)
    
    # Hourly KAMA (trend strength filter)
    price_series = pd.Series(close)
    # Efficiency Ratio (ER)
    change = abs(price_series.diff(10))
    volatility = price_series.diff().abs().rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume spike (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hour = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    
    # Start after KAMA warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0
            continue
            
        if np.isnan(ema1d_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0
            continue
            
        # Trend direction from daily EMA
        bullish = close[i] > ema1d_aligned[i]
        bearish = close[i] < ema1d_aligned[i]
        
        # KAMA slope for momentum confirmation
        kama_rising = kama[i] > kama[i-5]
        kama_falling = kama[i] < kama[i-5]
        
        # Enter long: bullish trend + rising KAMA + volume spike
        if bullish and kama_rising and vol_spike[i]:
            signals[i] = 0.20
        # Enter short: bearish trend + falling KAMA + volume spike
        elif bearish and kama_falling and vol_spike[i]:
            signals[i] = -0.20
        else:
            signals[i] = 0
    
    return signals