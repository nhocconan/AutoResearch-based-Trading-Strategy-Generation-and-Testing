#!/usr/bin/env python3
name = "12h_1d_Camarilla_R3S3_Breakout_1wTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels: R3, S3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot calculation: (H+L+C)/3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # R3 = H + 2*(Pivot - L)
    # S3 = L - 2*(H - Pivot)
    r3_1d = high_1d + 2.0 * (pivot_1d - low_1d)
    s3_1d = low_1d - 2.0 * (high_1d - pivot_1d)
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Simple 20-period EMA for weekly trend
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 12h volume filter: > 1.8x 30-period average (more selective)
    vol_ma_12h = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_filter = volume > 1.8 * vol_ma_12h
    
    # ATR for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    vol_regime = atr > pd.Series(atr).rolling(window=50, min_periods=50).mean().values  # high vol regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_12h[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R3 with weekly uptrend, volume, and high volatility
            if (close[i] > r3_1d_aligned[i] and close[i] > ema_1w_aligned[i] and 
                vol_filter[i] and vol_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < S3 with weekly downtrend, volume, and high volatility
            elif (close[i] < s3_1d_aligned[i] and close[i] < ema_1w_aligned[i] and 
                  vol_filter[i] and vol_regime[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA_1w (trend change) or volatility drops
            if close[i] < ema_1w_aligned[i] or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA_1w (trend change) or volatility drops
            if close[i] > ema_1w_aligned[i] or not vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA trend filter, volume confirmation, and volatility regime filter.
# Camarilla levels identify key support/resistance from daily price action.
# Breakout above R3 in uptrend (price > 20 EMA) or below S3 in downtrend captures momentum.
# Volume filter ensures institutional participation (>1.8x 30-period average).
# Volatility regime filter ensures trades only in high volatility periods (reduces whipsaw).
# Target: 20-40 trades/year to minimize fee drag. Position size 0.25 limits risk.