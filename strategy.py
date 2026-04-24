#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency and reduced fee drag.
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to capture institutional interest.
- Williams %R: 14-period for mean reversion signals.
- Entry: Long when Williams %R < -80 (oversold) AND 1d EMA34 bullish AND volume spike.
         Short when Williams %R > -20 (overbought) AND 1d EMA34 bearish AND volume spike.
- Exit: Opposite Williams %R level (Williams %R > -20 for long, Williams %R < -80 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown while minimizing fee churn.
- Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe.
This strategy captures mean reversion in the direction of the daily trend, with volume spikes confirming
institutional participation. Works in both bull and bear markets by only taking trades aligned with the 1d trend.
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
    
    # Calculate 4h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    willr = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50.0)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need enough bars for EMA34, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(willr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_willr = willr[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Williams %R < -80 (oversold) AND 1d EMA34 bullish (close > EMA)
                if curr_willr < -80.0 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R > -20 (overbought) AND 1d EMA34 bearish (close < EMA)
                elif curr_willr > -20.0 and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (overbought) OR loss of volume confirmation
            if curr_willr > -20.0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (oversold) OR loss of volume confirmation
            if curr_willr < -80.0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0