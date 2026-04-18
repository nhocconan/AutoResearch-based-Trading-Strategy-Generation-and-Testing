#!/usr/bin/env python3
"""
1h_RSI_Extremes_4hTrend_1dVol
Hypothesis: In 1-hour timeframe, RSI extremes (oversold/overbought) combined with 4-hour trend direction and daily volume spikes provide high-probability mean-reversion entries with trend filtering. Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend). Volume spikes confirm institutional interest. Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag.
"""

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
    
    # 4-hour trend: EMA50
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1-hour RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day volume spike: current volume > 2.0 x 20-day average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 20)  # Warmup period
    
    for i in range(start_idx, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_spike_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI oversold (<30) + 4h uptrend (price > EMA50) + volume spike
            if (rsi[i] < 30 and close[i] > ema50_4h_aligned[i] and vol_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) + 4h downtrend (price < EMA50) + volume spike
            elif (rsi[i] > 70 and close[i] < ema50_4h_aligned[i] and vol_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60) or 4h trend breaks
            if (rsi[i] >= 40 or close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60) or 4h trend breaks
            if (rsi[i] <= 60 or close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_Extremes_4hTrend_1dVol"
timeframe = "1h"
leverage = 1.0