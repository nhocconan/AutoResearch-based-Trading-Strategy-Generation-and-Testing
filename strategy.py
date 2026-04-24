#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume spike confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA trend filter and volume average.
- Williams %R(14): identifies overbought/oversold conditions for mean reversion.
- Entry: Long when Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.5 * 1d average volume.
         Short when Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.5 * 1d average volume.
- Exit: Opposite Williams %R signal (%R > -50 for long exit, %R < -50 for short exit).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures exhaustion moves in ranging markets.
- 1d EMA50 ensures we trade with the higher timeframe trend.
- Volume confirmation prevents false signals in low participation.
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate Williams %R(14) from 12h data
    wr_period = 14
    wr = williams_r(high, low, close, wr_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(wr_period, 20, 50)  # Need 14 for WR, 20 for volume MA, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Exit conditions: Williams %R mean reversion
        if position != 0:
            # Exit long: Williams %R rises above -50 (leaving oversold territory)
            if position == 1:
                if wr[i] > -50:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R falls below -50 (leaving overbought territory)
            elif position == -1:
                if wr[i] < -50:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R extreme with trend filter and volume confirmation
        if position == 0:
            # Williams %R signals
            wr_oversold = wr[i] < -80
            wr_overbought = wr[i] > -20
            
            # Trend filter: price relative to 1d EMA50
            price_above_ema = curr_close > ema_50_1d_aligned[i]
            price_below_ema = curr_close < ema_50_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            if wr_oversold and price_above_ema and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif wr_overbought and price_below_ema and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR14_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0