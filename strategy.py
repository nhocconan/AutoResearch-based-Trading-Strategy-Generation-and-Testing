#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h EMA trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction (bullish when price > EMA50, bearish when price < EMA50).
- Williams %R(14): Measures overbought/oversold levels on 6h chart.
- Entry: Long when Williams %R crosses above -80 (oversold bounce) AND 12h EMA50 trend bullish AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 (overbought rejection) AND 12h EMA50 trend bearish AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R level (long exit when %R crosses below -20, short exit when %R crosses above -80).
- Signal size: 0.25 discrete to minimize fee drag.
- Works in both bull and bear markets by aligning with 12h trend and fading extremes in 6h momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h volume average for confirmation (20-period)
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20_12h)
    
    # Calculate 6h Williams %R(14)
    williams_window = 14
    highest_high = pd.Series(high).rolling(window=williams_window, min_periods=williams_window).max().values
    lowest_low = pd.Series(low).rolling(window=williams_window, min_periods=williams_window).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(williams_window, 50)  # Need 14 for Williams %R, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20_12h_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams = williams_r[i]
        prev_williams = williams_r[i-1]
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        bullish_trend = curr_close > ema50_12h_aligned[i]
        bearish_trend = curr_close < ema50_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20_12h_aligned[i] if not np.isnan(vol_ma_20_12h_aligned[i]) else False
        
        # Williams %R crossovers
        crossed_above_oversold = prev_williams <= -80 and curr_williams > -80
        crossed_below_overbought = prev_williams >= -20 and curr_williams < -20
        
        # Exit conditions: opposite Williams %R level
        if position != 0:
            # Exit long: Williams %R crosses below -20 (overbought)
            if position == 1 and crossed_below_overbought:
                signals[i] = 0.0
                position = 0
                continue
            # Exit short: Williams %R crosses above -80 (oversold)
            elif position == -1 and crossed_above_oversold:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Williams %R crossover with trend and volume filters
        if position == 0:
            # Long: Williams %R crosses above -80 (oversold bounce) AND bullish trend AND volume confirmation
            long_condition = crossed_above_oversold and bullish_trend and volume_confirm
            
            # Short: Williams %R crosses below -20 (overbought rejection) AND bearish trend AND volume confirmation
            short_condition = crossed_below_overbought and bearish_trend and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0