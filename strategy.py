#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h for EMA50 trend direction, 1d for volume average and Williams %R calculation.
- Williams %R(14): identifies overbought/oversold conditions for mean reversion.
- Entry: Long when Williams %R < -80 (oversold) AND price > 12h EMA50 (uptrend) AND volume > 2.0 * 1d average volume.
         Short when Williams %R > -20 (overbought) AND price < 12h EMA50 (downtrend) AND volume > 2.0 * 1d average volume.
- Exit: Opposite Williams %R signal (Williams %R > -50 for longs, < -50 for shorts) to capture mean reversion swings.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures short-term exhaustion in trends.
- 12h EMA50 filter ensures trades align with intermediate-term trend.
- Volume confirmation prevents false signals in low-activity periods.
- Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
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
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 1d Williams %R(14) for mean reversion signals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need sufficient data for Williams %R
        return np.zeros(n)
    
    wr_14_1d = williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    wr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_14_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # Need 50 for EMA, 14 for Williams %R, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(wr_14_1d_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: Williams %R mean reversion (exit when momentum returns)
        if position != 0:
            # Exit long: Williams %R rises above -50 (momentum returning)
            if position == 1:
                if wr_14_1d_aligned[i] > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R falls below -50 (momentum returning)
            elif position == -1:
                if wr_14_1d_aligned[i] < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme + 12h trend + volume confirmation
        if position == 0:
            # Williams %R signals
            oversold = wr_14_1d_aligned[i] < -80
            overbought = wr_14_1d_aligned[i] > -20
            
            # Trend filter: price relative to 12h EMA50
            uptrend = curr_close > ema_50_12h_aligned[i]
            downtrend = curr_close < ema_50_12h_aligned[i]
            
            # Volume confirmation: current volume > 2.0 * 20-period average volume (aligned)
            volume_confirm = volume[i] > 2.0 * vol_ma_20_aligned[i]
            
            if oversold and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif overbought and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0