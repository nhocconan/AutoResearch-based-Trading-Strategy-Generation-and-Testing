#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 12h EMA200 Trend Filter + Volume Spike Confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h EMA200 for trend direction (bullish if close > EMA200, bearish if close < EMA200).
- Williams %R(14) extreme levels: Long when %R < -90 (oversold), Short when %R > -10 (overbought).
- Volume confirmation: Current volume > 2.0 * volume MA(20) on 6h chart.
- Exit: Close-based reversal - exit long when Williams %R crosses above -50,
        exit short when Williams %R crosses below -50 (mean reversion to midline).
- Signal size: 0.25 discrete to balance return and drawdown control.
Designed to work in both bull and bear markets by fading extremes in the direction of the 12h trend.
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
    
    # Get 12h data for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    # Calculate 12h EMA200 for trend filter
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high - lowest_low
    williams_r = np.where(rr != 0, ((highest_high - close) / rr) * -100, -50)
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate volume MA(20) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 200, 20, 14)  # Need enough bars for EMA200, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Williams %R < -90 (extreme oversold) AND 12h EMA200 bullish AND volume confirmed
            if curr_williams_r < -90.0 and curr_close > ema_12h_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -10 (extreme overbought) AND 12h EMA200 bearish AND volume confirmed
            elif curr_williams_r > -10.0 and curr_close < ema_12h_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R crosses above -50 (mean reversion)
            if curr_williams_r > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses below -50 (mean reversion)
            if curr_williams_r < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA200_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0