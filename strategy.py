#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when Williams %R < -80 (oversold) AND 1d trend bullish AND volume spike.
         Short when Williams %R > -20 (overbought) AND 1d trend bearish AND volume spike.
- Exit: Opposite Williams %R condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 4h timeframe.
Williams %R is a momentum oscillator that works well in ranging markets and catches reversals.
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
    
    # Calculate Williams %R on 4h (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    df_1d_close = df_1d['close'].values
    ema_34 = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_20_aligned)
    
    # Trend filter: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    trend = np.where(close > ema_34_aligned, 1, np.where(close < ema_34_aligned, -1, 0))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(trend[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        tr = trend[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish: Williams %R oversold (< -80) AND 1d trend bullish
                if wr < -80 and tr == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R overbought (> -20) AND 1d trend bearish
                elif wr > -20 and tr == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -50 (momentum fading) OR loss of volume confirmation
            if wr > -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -50 (momentum fading) OR loss of volume confirmation
            if wr < -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0