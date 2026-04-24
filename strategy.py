#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1-week EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 6h to balance trade frequency and signal quality.
- Williams %R(14): Long when < -80 (oversold), Short when > -20 (overbought).
- HTF: 1-week EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 6h volume > 2.0 * 20-period 6h volume MA to capture institutional interest.
- Entry: Long when Williams %R < -80 AND 1w EMA50 bullish AND volume spike.
         Short when Williams %R > -20 AND 1w EMA50 bearish AND volume spike.
- Exit: Williams %R reverts to -50 (mean reversion) OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy captures mean reversion moves in the direction of the weekly trend,
avoiding counter-trend trades. Volume spikes confirm institutional participation,
making it effective in both bull and bear markets by aligning with the higher timeframe trend.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low + 1e-10) * -100
    
    # Calculate 20-period 6h volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current 6h volume > 2.0 * 20-period 6h volume MA
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need enough bars for EMA50, Williams %R, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_williams_r = williams_r[i]
        
        if position == 0:
            # Check for extreme Williams %R signals with volume spike
            if volume_spike[i]:
                # Oversold long: Williams %R < -80 AND 1w EMA50 bullish (close > EMA)
                if curr_williams_r < -80 and close[i] > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Overbought short: Williams %R > -20 AND 1w EMA50 bearish (close < EMA)
                elif curr_williams_r > -20 and close[i] < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reverts to -50 OR loss of volume confirmation
            if curr_williams_r >= -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts to -50 OR loss of volume confirmation
            if curr_williams_r <= -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0