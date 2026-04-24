#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 12h EMA50 Trend + Volume Spike
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams %R(14): Extreme oversold < -80 for long, extreme overbought > -20 for short.
- Entry: Long when Williams %R crosses above -80 AND 12h EMA50 bullish AND volume > 1.8 * volume MA(20).
         Short when Williams %R crosses below -20 AND 12h EMA50 bearish AND volume > 1.8 * volume MA(20).
- Exit: Close-based reversal - exit long when Williams %R crosses above -50,
        exit short when Williams %R crosses below -50.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures mean reversion extremes aligned with the 12h trend, designed to work in both bull and bear markets by fading exhaustion while respecting trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 55, 20)  # Need enough bars for EMA50, Williams %R, and vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or (highest_high[i] - lowest_low[i]) == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        prev_wr = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Long: Williams %R crosses above -80 (from below) AND 12h EMA50 bullish AND volume confirmed
            if prev_wr <= -80 and curr_wr > -80 and curr_close > ema_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above) AND 12h EMA50 bearish AND volume confirmed
            elif prev_wr >= -20 and curr_wr < -20 and curr_close < ema_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R crosses above -50 (mean reversion complete)
            if prev_wr <= -50 and curr_wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses below -50 (mean reversion complete)
            if prev_wr >= -50 and curr_wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0