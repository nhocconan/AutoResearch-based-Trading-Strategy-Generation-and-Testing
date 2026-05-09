#!/usr/bin/env python3
"""
4h_ChaikinMoneyFlow_DCM_Breakout_1dTrend_Volume
Hypothesis: Chaikin Money Flow (CMF) measures institutional money flow.
Enter long when CMF > 0.15 and price breaks above Donchian Channel (20) upper band,
short when CMF < -0.15 and price breaks below lower band.
Use 1-day EMA50 as trend filter to ensure alignment with higher timeframe trend.
Volume confirmation via volume > 1.5x 20-period average.
Session filter: 08:00-20:00 UTC to avoid low liquidity periods.
Designed for low trade frequency (20-40/year) with high win rate by requiring
money flow confirmation, trend alignment, and volatility breakout.
Works in bull markets via long signals and bear markets via short signals.
"""

name = "4h_ChaikinMoneyFlow_DCM_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20)
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Chaikin Money Flow (20)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mf_multiplier = ((close - low) - (high - close)) / (high - low)
    # Replace division by zero or NaN with 0
    mf_multiplier = np.where((high - low) == 0, 0, mf_multiplier)
    mf_multiplier = np.where(np.isnan(mf_multiplier), 0, mf_multiplier)
    # Money Flow Volume = Money Flow Multiplier * Volume
    mf_volume = mf_multiplier * volume
    # CMF = 20-period sum of MFV / 20-period sum of Volume
    mf_volume_sum = pd.Series(mf_volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(volume_sum == 0, 0, mf_volume_sum / volume_sum)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA50 trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    trend_up = close > ema_50_1d_aligned
    trend_down = close < ema_50_1d_aligned
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(cmf[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or np.isnan(volume_filter[i]) or
            np.isnan(session_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: CMF > 0.15 + breakout above DC high + 1d uptrend + volume spike + session
            if cmf[i] > 0.15 and close[i] > dc_high[i] and trend_up[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: CMF < -0.15 + breakout below DC low + 1d downtrend + volume spike + session
            elif cmf[i] < -0.15 and close[i] < dc_low[i] and trend_down[i] and volume_filter[i] and session_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle or trend reversal
            dc_middle = (dc_high[i] + dc_low[i]) / 2
            if close[i] < dc_middle or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian middle or trend reversal
            dc_middle = (dc_high[i] + dc_low[i]) / 2
            if close[i] > dc_middle or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals