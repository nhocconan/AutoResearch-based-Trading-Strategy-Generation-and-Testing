#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R reversal with 1d EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams %R(14): Oversold < -80 for long, overbought > -20 for short.
- Volume: Current 4h volume > 1.8 * 20-period 4h volume MA to confirm momentum.
- Entry: Long when Williams %R crosses above -80 AND 1d EMA trend bullish AND volume spike.
         Short when Williams %R crosses below -20 AND 1d EMA trend bearish AND volume spike.
- Exit: Opposite Williams %R condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
Williams %R is a momentum oscillator that identifies overbought/oversold levels.
In strong trends (1d EMA filter), it can catch meaningful reversals with less whipsaw.
Works in both bull and bear markets by aligning with higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R on 4h
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 1d trend: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    trend_1d = np.where(df_1d_close > ema_34_1d, 1, np.where(df_1d_close < ema_34_1d, -1, 0))
    
    # Calculate 20-period volume MA on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 4h volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough bars for Williams %R and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        trend_val = trend_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Williams %R crosses above -80 (oversold reversal) AND 1d trend bullish
                if wr_prev <= -80 and wr > -80 and trend_val == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R crosses below -20 (overbought reversal) AND 1d trend bearish
                elif wr_prev >= -20 and wr < -20 and trend_val == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R crosses below -50 (momentum loss) OR loss of volume confirmation
            if wr_prev > -50 and wr < -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R crosses above -50 (momentum loss) OR loss of volume confirmation
            if wr_prev < -50 and wr > -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0