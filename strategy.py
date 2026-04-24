#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1w EMA200 trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA200 for trend direction (bullish if close > EMA200, bearish if close < EMA200).
- Williams %R(14) from 6h data: Long when %R crosses above -80 from oversold, Short when %R crosses below -20 from overbought.
- Entry: Only take longs in bullish trend (close > 1w EMA200) and shorts in bearish trend (close < 1w EMA200).
- Volume confirmation: Require volume > 1.5 * volume MA(20) to avoid low-quality breakouts.
- Exit: Reverse signal or when %R crosses opposite extreme (%R < -80 for longs, %R > -20 for shorts).
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures mean reversions in extreme conditions while respecting the weekly trend, working in both bull and bear markets by filtering counter-trend signals.
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Williams %R(14) from 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, 20)  # Need enough bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Bullish trend filter: close > 1w EMA200
            bullish_trend = curr_close > ema_1w_aligned[i]
            # Bearish trend filter: close < 1w EMA200
            bearish_trend = curr_close < ema_1w_aligned[i]
            
            # Long: Williams %R crosses above -80 from oversold AND bullish trend AND volume confirmed
            if (curr_wr > -80 and williams_r[i-1] <= -80 and bullish_trend and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from overbought AND bearish trend AND volume confirmed
            elif (curr_wr < -20 and williams_r[i-1] >= -20 and bearish_trend and vol_confirmed):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Williams %R crosses below -80 (mean reversion complete) or reverse signal
            if curr_wr < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Williams %R crosses above -20 (mean reversion complete) or reverse signal
            if curr_wr > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1wEMA200_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0