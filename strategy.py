#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d EMA200 trend filter and volume spike confirmation.
- Primary timeframe: 6h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA200 for trend direction (bullish if close > EMA200, bearish if close < EMA200).
- Williams %R(14): Long when %R crosses above -80 from oversold, Short when %R crosses below -20 from overbought.
- Volume confirmation: volume > 1.5 * volume MA(20) to filter weak breakouts.
- Exit: Close-based reversal - exit long when %R crosses below -50, exit short when %R crosses above -50.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures mean reversals in overextended moves while aligning with the daily trend,
designed to work in both bull and bear markets by fading extremes in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # Need enough bars for EMA200 and Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        prev_wr = williams_r[i-1] if i > 0 else -50
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Williams %R crosses above -80 from oversold AND 1d EMA200 bullish AND volume confirmed
            if prev_wr <= -80 and curr_wr > -80 and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought AND 1d EMA200 bearish AND volume confirmed
            elif prev_wr >= -20 and curr_wr < -20 and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R crosses below -50 (momentum weakening)
            if prev_wr >= -50 and curr_wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses above -50 (momentum weakening)
            if prev_wr <= -50 and curr_wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA200_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0