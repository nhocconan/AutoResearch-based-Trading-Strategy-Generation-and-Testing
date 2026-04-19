#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation
# Williams %R(14) identifies overbought/oversold conditions (> -20 short, < -80 long)
# 1d EMA(34) determines trend direction (only trade in direction of higher timeframe trend)
# 1d volume spike (>1.5x average) confirms conviction
# Designed to capture mean reversion within established trends
# Target: 20-40 trades/year to avoid fee drag
name = "6h_WilliamsR_1dTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.5x average
        vol_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above 1d EMA + volume
            if williams_r[i] < -80 and close[i] > ema_34_1d_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below 1d EMA + volume
            elif williams_r[i] > -20 and close[i] < ema_34_1d_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R returns to neutral (> -50) or reverse signal
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R returns to neutral (< -50) or reverse signal
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals