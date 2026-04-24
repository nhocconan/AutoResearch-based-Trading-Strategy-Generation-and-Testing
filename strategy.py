#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for EMA34 trend filter and volume average.
- Williams %R: momentum oscillator measuring overbought/oversold levels.
- Entry: Long when Williams %R crosses above -80 (oversold) AND price > 1d EMA34 AND volume > 1.5 * 20-period average volume.
         Short when Williams %R crosses below -20 (overbought) AND price < 1d EMA34 AND volume > 1.5 * 20-period average volume.
- Exit: Opposite Williams %R signal (long exits when crosses below -20, short exits when crosses above -80).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R works in ranging markets (mean reversion) and trends (pullbacks in direction of trend).
- Volume confirmation ensures participation.
- 1d EMA34 provides smooth trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, lookback=14):
    """Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100"""
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max()
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return wr.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34
        return np.zeros(n)
    
    ema34_1d = ema(df_1d['close'].values, 34)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average for confirmation
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)  # Avoid division by zero
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate Williams %R from 4h data (14-period)
    wr = williams_r(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 34, 20)  # Need 14 for Williams %R, 34 for 1d EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or
            np.isnan(wr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_wr = wr[i]
        prev_wr = wr[i-1]
        
        # Exit conditions: opposite Williams %R signal
        if position != 0:
            # Exit long: Williams %R crosses below -20 (overbought)
            if position == 1:
                if curr_wr < -20 and prev_wr >= -20:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: Williams %R crosses above -80 (oversold)
            elif position == -1:
                if curr_wr > -80 and prev_wr <= -80:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Williams %R signal with trend filter and volume confirmation
        if position == 0:
            # Williams %R crosses above -80 (oversold to upward momentum)
            wr_long_signal = curr_wr > -80 and prev_wr <= -80
            # Williams %R crosses below -20 (overbought to downward momentum)
            wr_short_signal = curr_wr < -20 and prev_wr >= -20
            
            # Trend filter: price vs 1d EMA34
            long_trend = curr_close > ema34_1d_aligned[i]
            short_trend = curr_close < ema34_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
            vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
            
            if wr_long_signal and long_trend and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif wr_short_signal and short_trend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_1dEMA34_TrendFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0