#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R with 1w trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend direction (EMA34), 1d for volume average.
- Williams %R(14): identifies overbought/oversold conditions.
- Entry: Long when Williams %R crosses above -80 from below AND price > 1w EMA34 (bullish trend) AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 from above AND price < 1w EMA34 (bearish trend) AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R signal (crosses below -20 for longs, above -80 for shorts) or trailing stop.
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures mean reversion in extended moves, filtered by weekly trend.
- Volume confirmation ensures legitimacy of the move.
- Works in both bull and bear markets by aligning with weekly trend direction.
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
    if n < 40:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Williams %R(14) on 6h data
    if len(prices) < 14:
        return np.zeros(n)
    
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 14)  # Need 34 for weekly EMA, 20 for volume MA, 14 for Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        prev_wr = wr[i-1]
        prev_close = close[i-1]
        
        # Exit conditions: Williams %R reversal signals
        if position != 0:
            # Exit long: Williams %R crosses below -20 from above
            if position == 1:
                if wr[i] < -20 and prev_wr >= -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -80 from below
            elif position == -1:
                if wr[i] > -80 and prev_wr <= -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend filter and volume confirmation
        if position == 0:
            # Williams %R signals
            wr_cross_up = wr[i] > -80 and prev_wr <= -80  # Cross above -80 from below
            wr_cross_down = wr[i] < -20 and prev_wr >= -20  # Cross below -20 from above
            
            # Trend filter: price vs weekly EMA34
            uptrend = curr_close > ema_34_1w_aligned[i]
            downtrend = curr_close < ema_34_1w_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
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

name = "6h_WilliamsR14_1wEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0