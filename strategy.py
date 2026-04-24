#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA50 trend direction and volume average.
- Williams %R(14): identifies overbought/oversold conditions.
- Entry: Long when Williams %R crosses above -80 (oversold recovery) AND price > 1d EMA50 (uptrend) AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 (overbought decline) AND price < 1d EMA50 (downtrend) AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R signal (crosses below -20 for longs, above -80 for shorts).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures mean-reversion bounces in ranging markets.
- Trend filter ensures trades align with higher timeframe momentum.
- Volume confirmation reduces false signals.
- Works in bull markets (long bias) and bear markets (short bias) via 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=period, min_periods=period).max()
    lowest_low = low_series.rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close_series) / (highest_high - lowest_low)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 6h Williams %R(14)
    if len(prices) < 14:
        return np.zeros(n)
    
    wr_14 = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need 50 for EMA50, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(wr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = wr_14[i]
        prev_wr = wr_14[i-1]
        curr_volume = volume[i]
        
        # Exit conditions: Williams %R crosses opposite threshold
        if position != 0:
            # Exit long: Williams %R crosses below -20 (overbought)
            if position == 1:
                if prev_wr > -20 and curr_wr <= -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -80 (oversold)
            elif position == -1:
                if prev_wr < -80 and curr_wr >= -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volume confirmation
        if position == 0:
            # Williams %R reversal signals
            wr_cross_up = prev_wr <= -80 and curr_wr > -80   # Cross above -80 (oversold recovery)
            wr_cross_down = prev_wr >= -20 and curr_wr < -20  # Cross below -20 (overbought decline)
            
            # Trend filter: price vs 1d EMA50
            uptrend = curr_close > ema_50_1d_aligned[i]
            downtrend = curr_close < ema_50_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            if wr_cross_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif wr_cross_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0