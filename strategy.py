#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA50 trend filter + volume spike confirmation
- Williams %R(14) identifies overbought/oversold conditions on 6h chart
- 1d EMA50 provides stronger trend filter than shorter EMAs for 6h timeframe
- Volume spike (2.0x 20-period MA) confirms institutional participation
- Discrete position sizing (0.25) minimizes fee churn
- Target: 12-25 trades/year per symbol (~50-100 total over 4 years)
- Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend)
- Williams %R is mean-reverting but trend-filtered, reducing false signals in strong trends
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
    
    # Get 6h data for Williams %R calculation
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 6h: (Highest High - Close) / (Highest High - Lowest Low) * -100
    def williams_r(high_arr, low_arr, close_arr, window=14):
        highest_high = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        lowest_low = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        wr = -100 * (highest_high - close_arr) / (highest_high - lowest_low)
        # Handle division by zero (when high == low)
        wr = np.where((highest_high - lowest_low) == 0, -50, wr)
        return wr
    
    wr_6h = williams_r(high_6h, low_6h, close_6h, 14)
    
    # Calculate EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 6h
    volume_ma_6h = pd.Series(df_6h['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    wr_6h_aligned = align_htf_to_ltf(prices, df_6h, wr_6h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(wr_6h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        wr = wr_6h_aligned[i]
        ema_trend = ema50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = df_6h['volume'].values[i]  # volume from 6h data
        price = close[i]
        
        if position == 0:
            # Long: Williams %R oversold (< -80) + price > 1d EMA50 (uptrend) + volume spike
            if wr < -80 and price > ema_trend and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) + price < 1d EMA50 (downtrend) + volume spike
            elif wr > -20 and price < ema_trend and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or reverse signal
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or reverse signal
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_1dEMA50_VolumeSpike"
timeframe = "6h"
leverage = 1.0