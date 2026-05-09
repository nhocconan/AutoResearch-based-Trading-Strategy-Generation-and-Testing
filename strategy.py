#!/usr/bin/env python3
name = "1h_4hTrend_1dCci_MeanReversion"
timeframe = "1h"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 trend filter
    ema_20_4h = pd.Series(df_4h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for CCI mean reversion signal
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 14-period CCI on daily data
    typical_price = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    sma_tp = pd.Series(typical_price).rolling(window=14, min_periods=14).mean().values
    mean_dev = pd.Series(typical_price).rolling(window=14, min_periods=14).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    cci_14 = (typical_price - sma_tp) / (0.015 * mean_dev)
    cci_14_aligned = align_htf_to_ltf(prices, df_1d, cci_14)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Volume filter: current volume > 1.5x 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(cci_14_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend + CCI oversold (< -100) + volume
            if (close[i] > ema_20_4h_aligned[i]) and (cci_14_aligned[i] < -100) and volume_filter[i]:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + CCI overbought (> 100) + volume
            elif (close[i] < ema_20_4h_aligned[i]) and (cci_14_aligned[i] > 100) and volume_filter[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: CCI returns to neutral (> -50) or trend reversal
            if cci_14_aligned[i] > -50 or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: CCI returns to neutral (< 50) or trend reversal
            if cci_14_aligned[i] < 50 or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals