#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams %R(14) calculated on 4h data for overbought/oversold signals.
- Entry: Long when Williams %R crosses above -80 (oversold) AND 1d EMA50 bullish AND volume > 1.8 * volume MA(20).
         Short when Williams %R crosses below -20 (overbought) AND 1d EMA50 bearish AND volume > 1.8 * volume MA(20).
- Exit: Close-based reversal - exit long when Williams %R crosses below -20,
        exit short when Williams %R crosses above -80.
- Signal size: 0.25 discrete to balance return and drawdown.
Uses Williams %R for mean reversion in ranging markets combined with trend filter to avoid counter-trend trades.
Works in both bull and bear markets by only taking trades in direction of 1d trend.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Williams %R(14) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 55)  # Need enough bars for Williams %R and EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams = williams_r[i]
        prev_williams = williams_r[i-1]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.8x threshold)
            vol_confirmed = curr_volume > 1.8 * vol_ma[i]
            
            # Long: Williams %R crosses above -80 (oversold) AND 1d EMA50 bullish AND volume confirmed
            if curr_williams > -80 and prev_williams <= -80 and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (overbought) AND 1d EMA50 bearish AND volume confirmed
            elif curr_williams < -20 and prev_williams >= -20 and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R crosses below -20 (overbought)
            if curr_williams < -20 and prev_williams >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses above -80 (oversold)
            if curr_williams > -80 and prev_williams <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0