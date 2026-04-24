#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channel: 20-bar high/low on 12h timeframe.
- Entry: Long when price breaks above 20-period high AND 1d EMA50 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below 20-period low AND 1d EMA50 bearish AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 20-period low,
        exit short when price crosses above 20-period high.
- Signal size: 0.25 discrete to control drawdown and minimize fee churn.
Designed to capture medium-term trends in both bull and bear markets via trend filter.
Donchian breakouts provide objective entry/exit levels with proven efficacy across cycles.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channel (20-period) on 12h data
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50, donchian_period, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above Donchian high AND 1d EMA50 bullish AND volume confirmed
            if curr_high > highest_high[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low AND 1d EMA50 bearish AND volume confirmed
            elif curr_low < lowest_low[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below Donchian low (trend reversal)
            if curr_low < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above Donchian high (trend reversal)
            if curr_high > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0