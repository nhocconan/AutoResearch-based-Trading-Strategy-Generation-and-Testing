#!/usr/bin/env python3
"""
Hypothesis: 1h Donchian breakout with 4h trend filter and volume spike confirmation.
- Primary timeframe: 1h for entries/exits.
- HTF: 4h Donchian(20) for trend direction (bullish if close > upper, bearish if close < lower).
- Volume: Current 1h volume > 1.8 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above 1h Donchian(20) upper AND 4h trend bullish AND volume spike.
         Short when price breaks below 1h Donchian(20) lower AND 4h trend bearish AND volume spike.
- Exit: Opposite Donchian breakout or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Donchian breakouts capture strong momentum moves, and the 4h filter ensures we trade with the higher timeframe trend.
Volume spike confirms institutional participation, reducing false signals in ranging markets.
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
    
    # Calculate 1h Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period) for trend
    df_4h_high = df_4h['high'].values
    df_4h_low = df_4h['low'].values
    df_4h_close = df_4h['close'].values
    
    period20_high_4h = pd.Series(df_4h_high).rolling(window=20, min_periods=20).max().values
    period20_low_4h = pd.Series(df_4h_low).rolling(window=20, min_periods=20).min().values
    
    # 4h trend: 1 if bullish (close > upper), -1 if bearish (close < lower), 0 otherwise
    trend_4h = np.where(df_4h_close > period20_high_4h, 1, np.where(df_4h_close < period20_low_4h, -1, 0))
    
    # Calculate 20-period volume MA on 4h for volume confirmation
    df_4h_volume = df_4h['volume'].values
    vol_ma_4h = pd.Series(df_4h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Volume confirmation: current 1h volume > 1.8 * 20-period 4h volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_4h_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # Need enough bars for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(trend_4h_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        trend_val = trend_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above 1h Donchian upper AND 4h trend bullish
                if curr_high > period20_high[i] and trend_val == 1:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below 1h Donchian lower AND 4h trend bearish
                elif curr_low < period20_low[i] and trend_val == -1:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below 1h Donchian lower OR loss of volume confirmation
            if curr_low < period20_low[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 1h Donchian upper OR loss of volume confirmation
            if curr_high > period20_high[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_DonchianBreakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0