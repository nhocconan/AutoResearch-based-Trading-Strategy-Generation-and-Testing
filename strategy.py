#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA trend filter and volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 12h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when Williams %R < -80 (oversold) AND 1d EMA trend bullish AND volume spike.
         Short when Williams %R > -20 (overbought) AND 1d EMA trend bearish AND volume spike.
- Exit: Opposite Williams %R condition or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams %R is a momentum oscillator that works well in ranging markets and captures reversals.
Combined with EMA trend filter and volume confirmation, it should avoid false signals in strong trends.
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
    
    # Calculate Williams %R on 12h (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close) / (period14_high - period14_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((period14_high - period14_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_34 = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_20 = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Trend filter: 1 if bullish (close > EMA34), -1 if bearish (close < EMA34), 0 otherwise
    trend_filter = np.where(close > ema_34_aligned, 1, np.where(close < ema_34_aligned, -1, 0))
    
    # Volume confirmation: current 12h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Need enough bars for Williams %R and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(trend_filter[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        trend = trend_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish: Williams %R oversold (< -80) AND trend bullish
                if wr < -80 and trend == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R overbought (> -20) AND trend bearish
                elif wr > -20 and trend == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R rises above -50 (exit oversold) OR loss of volume confirmation
            if wr > -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R falls below -50 (exit overbought) OR loss of volume confirmation
            if wr < -50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0