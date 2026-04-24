#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 12h trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA trend direction.
- Williams %R(14): identifies overbought/oversold conditions (long when %R crosses above -80 from below, short when crosses below -20 from above).
- Entry: Long when Williams %R crosses above -80 AND price > 12h EMA50 AND volume > 1.5 * 6h average volume.
         Short when Williams %R crosses below -20 AND price < 12h EMA50 AND volume > 1.5 * 6h average volume.
- Exit: Opposite Williams %R signal (%R crosses below -50 for long exit, above -50 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures mean reversions in ranging markets.
- 12h EMA50 ensures trading with the higher timeframe trend.
- Volume confirmation reduces false signals.
- Works in both bull and bear markets by adapting to trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R with proper min_periods."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA(50)
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h average volume for confirmation (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R(14) on 6h data
    wr_14 = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(wr_14[i]) or np.isnan(wr_14[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = wr_14[i]
        prev_wr = wr_14[i-1]
        curr_volume = volume[i]
        
        # Exit conditions: Williams %R mean reversion
        if position != 0:
            # Exit long: WR crosses below -50 (return from oversold)
            if position == 1:
                if prev_wr > -50 and curr_wr <= -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: WR crosses above -50 (return from overbought)
            elif position == -1:
                if prev_wr < -50 and curr_wr >= -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend and volume filter
        if position == 0:
            # Williams %R signals: crossing extremes
            wr_cross_up_80 = prev_wr <= -80 and curr_wr > -80   # Oversold bounce
            wr_cross_down_20 = prev_wr >= -20 and curr_wr < -20  # Overbought rejection
            
            # Trend filter: price vs 12h EMA50
            uptrend = curr_close > ema_50_12h_aligned[i]
            downtrend = curr_close < ema_50_12h_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
            
            if wr_cross_up_80 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif wr_cross_down_20 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_12hEMA50_TrendFilter_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0