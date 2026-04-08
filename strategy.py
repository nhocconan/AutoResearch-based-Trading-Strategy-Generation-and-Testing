#!/usr/bin/env python3
"""
1h Volume Spike + 4h Trend + Daily Volatility Regime Filter
Hypothesis: In strong trends (4h EMA21), 1h volume spikes above 2x average signal momentum continuation.
Only trade during low volatility regimes (daily ATR percentile < 50%) to avoid whipsaws.
Long when price > 4h EMA21, short when price < 4h EMA21. Volume spike confirms institutional interest.
Session filter (08-20 UTC) reduces noise. Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_volume_spike_4h_trend_daily_vol_regime_v1"
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
    
    # 4h EMA21 for trend
    df_4h = get_htf_data(prices, '4h')
    ema_4h_21 = df_4h['close'].ewm(span=21, adjust=False).mean().values
    ema_4h_21_1h = align_htf_to_ltf(prices, df_4h, ema_4h_21)
    
    # Daily ATR percentile for volatility regime (<50% = low vol)
    df_1d = get_htf_data(prices, '1d')
    atr_1d = pd.DataFrame({
        'high': df_1d['high'],
        'low': df_1d['low'],
        'close': df_1d['close']
    })
    atr_1d['tr'] = np.maximum(
        atr_1d['high'] - atr_1d['low'],
        np.maximum(
            abs(atr_1d['high'] - atr_1d['close'].shift(1)),
            abs(atr_1d['low'] - atr_1d['close'].shift(1))
        )
    )
    atr_1d_val = atr_1d['tr'].rolling(window=14, min_periods=14).mean().values
    atr_percentile = pd.Series(atr_1d_val).rolling(window=100, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_1h = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # 1h volume spike (>2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_4h_21_1h[i]) or np.isnan(atr_percentile_1h[i]) or 
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in low volatility regime (ATR percentile < 50)
        if atr_percentile_1h[i] >= 50:
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Long: price above 4h EMA21 + volume spike
        if close[i] > ema_4h_21_1h[i] and vol_spike[i]:
            signals[i] = 0.20
        # Short: price below 4h EMA21 + volume spike
        elif close[i] < ema_4h_21_1h[i] and vol_spike[i]:
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals