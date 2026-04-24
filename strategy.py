#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R with 1d EMA trend filter and volume spike confirmation.
- Primary timeframe: 12h for entries/exits.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Williams %R(14): measures overbought/oversold levels (-80 to -20 = oversold, -20 to 0 = overbought).
- Volume: Current 12h volume > 1.5 * 20-period volume MA to confirm breakout strength.
- Entry: Long when Williams %R crosses above -80 (exiting oversold) AND 1d EMA trend bullish AND volume spike.
         Short when Williams %R crosses below -20 (exiting overbought) AND 1d EMA trend bearish AND volume spike.
- Exit: Opposite Williams %R level (-20 for long, -80 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
Williams %R works in both bull and bear markets by capturing mean reversions from extremes while respecting the higher-timeframe trend.
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
    
    # Calculate Williams %R on 12h
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period14_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    period14_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (period14_high - close) / (period14_high - period14_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where(period14_high == period14_low, -50, williams_r)
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend
    df_1d_close = df_1d['close'].values
    ema_34_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Need enough bars for Williams %R and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        williams_val = williams_r[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Determine trend direction: 1 for bullish (close > EMA34), -1 for bearish (close < EMA34)
        trend_dir = 1 if curr_close > ema_trend else -1
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish entry: Williams %R crosses above -80 (exiting oversold) AND bullish trend
                if i > start_idx and williams_r[i-1] <= -80 and williams_val > -80 and trend_dir == 1:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses below -20 (exiting overbought) AND bearish trend
                elif i > start_idx and williams_r[i-1] >= -20 and williams_val < -20 and trend_dir == -1:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reaches -20 (overbought) OR loss of volume confirmation
            if williams_val >= -20 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reaches -80 (oversold) OR loss of volume confirmation
            if williams_val <= -80 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0