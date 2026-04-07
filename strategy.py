#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h 4h/1d Trend Filter with Volume Spike Entry
# Hypothesis: Use 4h EMA trend direction and 1d volatility regime to filter entries,
# entering on 1h pullbacks with volume spikes. Works in bull (buy pullbacks in uptrend)
# and bear (sell rallies in downtrend) by following higher timeframe trend.
# Volume spike confirms institutional interest. Target: 15-35 trades/year.

name = "1h_4h1d_trend_volume_spike_v1"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for volatility regime (ATR-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1h ATR for entry timing
    tr1h = high - low
    tr2h = np.abs(high - np.roll(close, 1))
    tr3h = np.abs(low - np.roll(close, 1))
    tr1h[0] = high[0] - low[0]
    tr2h[0] = np.abs(high[0] - close[0])
    tr3h[0] = np.abs(low[0] - close[0])
    trh = np.maximum(tr1h, np.maximum(tr2h, tr3h))
    atr_10_1h = pd.Series(trh).rolling(window=10, min_periods=10).mean().values
    
    # Volume filter: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_10_1h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 4h EMA
        trend_up = close[i] > ema_20_4h_aligned[i]
        trend_down = close[i] < ema_20_4h_aligned[i]
        
        # Volatility regime: only trade when volatility is elevated (above average)
        vol_regime = atr_10_1h[i] > 0.5 * atr_14_1d_aligned[i]
        
        # Long entry: uptrend + pullback to EMA + volume spike
        if trend_up and vol_regime and vol_spike[i]:
            if close[i] <= ema_20_4h_aligned[i] + 0.5 * atr_10_1h[i]:
                signals[i] = 0.20
        
        # Short entry: downtrend + rally to EMA + volume spike
        elif trend_down and vol_regime and vol_spike[i]:
            if close[i] >= ema_20_4h_aligned[i] - 0.5 * atr_10_1h[i]:
                signals[i] = -0.20
    
    return signals