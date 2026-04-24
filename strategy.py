#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1w trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w for trend direction (EMA34) and 1d for volume average.
- Williams %R(14): identifies overbought/oversold conditions for mean reversion.
- Entry: Long when Williams %R crosses above -80 from below AND price > 1w EMA34 (bullish trend) AND volume > 1.5 * 1d average volume.
         Short when Williams %R crosses below -20 from above AND price < 1w EMA34 (bearish trend) AND volume > 1.5 * 1d average volume.
- Exit: Opposite Williams %R signal (cross above -20 for long exit, cross below -80 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures short-term exhaustion in trending markets.
- 1w EMA34 filter ensures trades align with major trend, reducing whipsaws in sideways/choppy markets.
- Volume confirmation validates the strength of the reversal signal.
- Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_r(high, low, close, period):
    """Calculate Williams %R with proper min_periods."""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    wr = wr.replace([np.inf, -np.inf], np.nan)
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Williams %R(14) from 6h data
    wr_period = 14
    wr_values = williams_r(high, low, close, wr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(wr_period, 34, 20)  # Need 14 for WR, 34 for 1w EMA, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(wr_values[i]) or np.isnan(wr_values[i-1])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_wr = wr_values[i]
        prev_wr = wr_values[i-1]
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Exit conditions: opposite Williams %R signal
        if position != 0:
            # Exit long: Williams %R crosses above -20 (overbought)
            if position == 1:
                if prev_wr < -20 and curr_wr >= -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses below -80 (oversold)
            elif position == -1:
                if prev_wr > -80 and curr_wr <= -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R reversal with trend and volume filter
        if position == 0:
            # Williams %R reversal signals
            wr_cross_up_80 = prev_wr < -80 and curr_wr >= -80  # Cross above -80 (oversold)
            wr_cross_down_20 = prev_wr > -20 and curr_wr <= -20  # Cross below -20 (overbought)
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Trend filter: price relative to 1w EMA34
            price_above_trend = curr_close > ema_34_1w_aligned[i]
            price_below_trend = curr_close < ema_34_1w_aligned[i]
            
            if wr_cross_up_80 and price_above_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif wr_cross_down_20 and price_below_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR14_1wTrend_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0