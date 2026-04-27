#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and volume confirmation.
- In sideways markets, RSI extremes (>80 or <20) offer high-probability mean reversion
- 4h EMA50 filters for trend direction: only take longs above EMA50, shorts below
- Volume spike (>1.5x 20-period average) confirms participation
- Target: 20-30 trades/year to avoid fee drag
- Uses discrete position sizing (0.20) to minimize churn
- Session filter (08-20 UTC) reduces noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_4h_50 = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_50_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_50)
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_rsi(gain, loss, period=14):
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        if len(gain) < period:
            return np.full_like(gain, 50.0)  # neutral RSI when insufficient data
        
        # First average is simple mean
        avg_gain[period-1] = np.mean(gain[1:period])
        avg_loss[period-1] = np.mean(loss[1:period])
        
        # Wilder's smoothing
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = wilders_rsi(gain, loss, 14)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for all indicators
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_4h_50_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: RSI < 20 (oversold) + price above 4h EMA50 + volume spike
            if (rsi[i] < 20 and 
                close[i] > ema_4h_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: RSI > 80 (overbought) + price below 4h EMA50 + volume spike
            elif (rsi[i] > 80 and 
                  close[i] < ema_4h_50_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 60 (overbought) or price below 4h EMA50
            if (rsi[i] > 60 or 
                close[i] < ema_4h_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 40 (oversold) or price above 4h EMA50
            if (rsi[i] < 40 or 
                close[i] > ema_4h_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI20_80_EMA50Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0