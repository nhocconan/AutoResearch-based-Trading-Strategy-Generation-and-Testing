#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for EMA50 trend direction.
- Donchian breakout: Long when price > upper band (20-period high), short when price < lower band (20-period low).
- Volume confirmation: current volume > 1.5 * 20-period average volume.
- Exit: Opposite Donchian breakout or trailing stop via signal=0 when price crosses 10-period EMA in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by 1w EMA50).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    if len(close) < 20:
        return np.zeros(n)
    
    # Upper band: 20-period high, Lower band: 20-period low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema50_1w = ema(df_1w['close'].values, 50)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(volume) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Need 20 for Donchian/volume, 50 for EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirm = curr_volume > 1.5 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = curr_close > upper_band[i]
        breakout_down = curr_close < lower_band[i]
        
        # Trend filter: price > 1w EMA50 for long bias, price < 1w EMA50 for short bias
        long_trend = curr_close > ema50_1w_aligned[i]
        short_trend = curr_close < ema50_1w_aligned[i]
        
        # Exit conditions
        if position != 0:
            # Exit long: price breaks below lower band OR crosses below 10-period EMA
            if position == 1:
                if breakout_down or curr_close < ema50_1w_aligned[i]:  # Simple trend-based exit
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above upper band OR crosses above 10-period EMA
            elif position == -1:
                if breakout_up or curr_close > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with trend filter and volume confirmation
        if position == 0:
            # Long: breakout above upper band AND long trend bias AND volume confirmation
            long_condition = breakout_up and long_trend and volume_confirm
            
            # Short: breakout below lower band AND short trend bias AND volume confirmation
            short_condition = breakout_down and short_trend and volume_confirm
            
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

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0