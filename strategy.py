#!/usr/bin/env python3
name = "6h_BollingerTrend_Reversal"
timeframe = "6h"
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
    
    # Get 1-day data for Bollinger Bands and trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period Bollinger Bands on 1D
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper = sma_20 + 2.0 * std_20
    lower = sma_20 - 2.0 * std_20
    
    # 50-period EMA for trend on 1D
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 6h timeframe
    upper_6h = align_htf_to_ltf(prices, df_1d, upper)
    lower_6h = align_htf_to_ltf(prices, df_1d, lower)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Bollinger Bandwidth for volatility filter
    bb_width = (upper - lower) / sma_20
    bb_width_6h = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(upper_6h[i]) or np.isnan(lower_6h[i]) or np.isnan(ema_50_6h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price near lower BB + uptrend + volume confirmation + low volatility
            if (close[i] <= lower_6h[i] * 1.02) and (close[i] > ema_50_6h[i]) and vol_confirm[i] and (bb_width_6h[i] < 0.05):
                signals[i] = 0.25
                position = 1
            # Short: Price near upper BB + downtrend + volume confirmation + low volatility
            elif (close[i] >= upper_6h[i] * 0.98) and (close[i] < ema_50_6h[i]) and vol_confirm[i] and (bb_width_6h[i] < 0.05):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses above SMA or trend breaks
            if (close[i] >= sma_20[-1] if not np.isnan(sma_20[-1]) else upper_6h[i]) or (close[i] <= ema_50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses below SMA or trend breaks
            if (close[i] <= sma_20[-1] if not np.isnan(sma_20[-1]) else lower_6h[i]) or (close[i] >= ema_50_6h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals