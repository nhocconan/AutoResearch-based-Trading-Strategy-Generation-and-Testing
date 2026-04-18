#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_TrendFilter_v1
Hypothesis: Camarilla pivot levels from daily timeframe provide key support/resistance.
Long when price breaks above R1 with volume confirmation and price above weekly EMA34 (bullish bias).
Short when price breaks below S1 with volume confirmation and price below weekly EMA34 (bearish bias).
Trend filter avoids trading against the higher timeframe trend. Target: 25-40 trades/year.
Works in bull markets via breakouts and in bear markets via short breakdowns, avoiding range whipsaws.
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (based on previous day)
    R1 = np.full_like(high_1d, np.nan)
    S1 = np.full_like(low_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_ = prev_high - prev_low
        
        if range_ > 0:
            R1[i] = prev_close + 1.1 * range_ / 12
            S1[i] = prev_close - 1.1 * range_ / 12
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to 4h timeframe
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34) + 1  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_1w_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price relative to weekly EMA34
        bullish_bias = close[i] > ema_1w_4h[i]
        bearish_bias = close[i] < ema_1w_4h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in bullish bias
            if close[i] > R1_4h[i] and vol_confirm and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in bearish bias
            elif close[i] < S1_4h[i] and vol_confirm and bearish_bias:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1
            if close[i] < S1_4h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1
            if close[i] > R1_4h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0