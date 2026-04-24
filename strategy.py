#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA trend direction, 1d for volume spike filter.
- Donchian breakout provides clear entry/exit levels with proven edge in crypto.
- EMA(50) on 12h filters for trend alignment (avoid counter-trend breakouts).
- Volume spike (>1.5x 20-period MA) confirms breakout strength.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull markets via trend-following breakouts and bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 1d data for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d (for volume spike confirmation)
    # We'll use 1d volume MA but apply it to 4h bars via alignment
    volume_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA(50) and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50 = ema_50_12h_aligned[i]
        vol_ma = volume_ma_1d_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Volume confirmation: current 4h volume > 1.5 * 1d volume MA
        # Note: Comparing 4h volume to 1d volume MA requires scaling
        # Approximate: 1d volume ≈ 6 * 4h volume (since 1d = 6 * 4h bars)
        volume_spike = volume[i] > (1.5 * vol_ma / 6.0)
        
        if position == 0:
            # Check for entry signals
            if volume_spike:
                # Bullish breakout: price closes above upper Donchian AND above 12h EMA(50)
                if curr_close > upper and curr_close > ema_50:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian AND below 12h EMA(50)
                elif curr_close < lower and curr_close < ema_50:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below lower Donchian
            if curr_close < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above upper Donchian
            if curr_close > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0