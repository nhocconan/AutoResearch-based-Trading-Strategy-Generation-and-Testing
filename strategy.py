#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1h RSI Pullback with 4h Trend and 1d Volume Confirmation
# Hypothesis: In trending markets (4h), pullbacks to RSI(30) on 1h offer high-probability
# entries when confirmed by above-average 1d volume. Works in bull/bear by following 4h trend.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
# Uses 4h for trend direction, 1d for volume filter, 1h for entry timing.

name = "1h_rsi_pullback_4h_trend_1d_volume_v1"
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
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(vol_avg_20_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: 1d volume > 20-day average
        vol_ok = volume[i] > vol_avg_20_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 or trend changes
            if rsi[i] >= 70 or close[i] < ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short position
            # Exit: RSI < 30 or trend changes
            if rsi[i] <= 30 or close[i] > ema_20_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Pullback entries in direction of 4h trend with volume confirmation
            if vol_ok:
                if close[i] > ema_20_4h_aligned[i]:  # Uptrend
                    if rsi[i] <= 30:  # Pullback to oversold
                        position = 1
                        signals[i] = 0.20
                else:  # Downtrend
                    if rsi[i] >= 70:  # Pullback to overbought
                        position = -1
                        signals[i] = -0.20
    
    return signals