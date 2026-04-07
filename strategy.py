#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h Volume Spike + 4h Trend + 1d Volatility Filter
# Hypothesis: In 1h timeframe, volume spikes indicate institutional participation.
# We combine with 4h EMA for trend direction and 1d ATR to filter low volatility periods.
# This works in bull markets (trend following) and bear markets (mean reversion during volatility spikes).
# Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year (60-150 over 4 years).
name = "1h_volume_spike_4h_trend_1d_vol_filter_v1"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h EMA(34) for trend
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=34, adjust=False).mean().values
    ema_4h_1h = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.inf  # First period has no previous close
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_1h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma * 2.0)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if required data not available
        if (np.isnan(ema_4h_1h[i]) or np.isnan(atr_1d_1h[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 4h EMA or volatility too low
            if close[i] < ema_4h_1h[i] or atr_1d_1h[i] < np.nanpercentile(atr_1d_1h[:i+1], 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price crosses above 4h EMA or volatility too low
            if close[i] > ema_4h_1h[i] or atr_1d_1h[i] < np.nanpercentile(atr_1d_1h[:i+1], 20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Maintain short position
        else:  # Flat, look for entry
            # Require volume spike and sufficient volatility
            if vol_spike[i] and atr_1d_1h[i] >= np.nanpercentile(atr_1d_1h[:i+1], 20):
                # Long: price above 4h EMA (uptrend)
                if close[i] > ema_4h_1h[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: price below 4h EMA (downtrend)
                elif close[i] < ema_4h_1h[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals