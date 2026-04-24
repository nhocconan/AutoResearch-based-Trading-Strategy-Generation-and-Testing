#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 1d EMA34 Trend + Volume Spike (revisited with optimized parameters)
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for EMA34 trend filter and Williams %R calculation.
- Williams %R(14): Overbought > -20, Oversold < -80.
- Entry: Long when Williams %R crosses above -80 (oversold bounce) AND price > 1d EMA34 AND volume > 2.0 * 20-period average volume.
         Short when Williams %R crosses below -20 (overbought rejection) AND price < 1d EMA34 AND volume > 2.0 * 20-period average volume.
- Exit: Opposite Williams %R cross (Long exit: cross below -50; Short exit: cross above -50).
- Signal size: 0.25 discrete to minimize fee drag.
- Williams %R captures mean reversion extremes; EMA34 filter ensures trades follow higher timeframe trend; volume spike confirms conviction.
- Works in ranging markets (mean reversion at extremes) and avoids counter-trend trades in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema(values, period):
    """Calculate Exponential Moving Average with proper min_periods."""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def williams_r(high, low, close, period):
    """Calculate Williams %R."""
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
    
    # Calculate 1d Williams %R(14) and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need sufficient data for EMA34 and Williams %R
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema34_1d = ema(df_1d['close'].values, 34)
    
    # 1d Williams %R(14)
    wr_1d = williams_r(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    
    # Align indicators to 6h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    wr_1d_aligned = align_htf_to_ltf(prices, df_1d, wr_1d)
    
    # Calculate 1d volume average for confirmation (20-period)
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(wr_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Williams %R crossovers
        if i > 0:
            wr_prev = wr_1d_aligned[i-1]
            wr_curr = wr_1d_aligned[i]
            
            # Long entry: Williams %R crosses above -80 (oversold bounce)
            long_cross_up = wr_prev <= -80 and wr_curr > -80
            
            # Short entry: Williams %R crosses below -20 (overbought rejection)
            short_cross_down = wr_prev >= -20 and wr_curr < -20
            
            # Long exit: Williams %R crosses below -50
            long_exit = wr_prev > -50 and wr_curr <= -50
            
            # Short exit: Williams %R crosses above -50
            short_exit = wr_prev < -50 and wr_curr >= -50
        else:
            long_cross_up = False
            short_cross_down = False
            long_exit = False
            short_exit = False
        
        # Volume confirmation: current volume > 2.0 * 20-period average volume
        volume_confirm = curr_volume > 2.0 * vol_ma_20_1d_aligned[i] if not np.isnan(vol_ma_20_1d_aligned[i]) else False
        
        # Exit conditions
        if position != 0:
            # Exit long: Williams %R crosses below -50
            if position == 1 and long_exit:
                signals[i] = 0.0
                position = 0
                continue
            # Exit short: Williams %R crosses above -50
            elif position == -1 and short_exit:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: Williams %R with trend filter and volume confirmation
        if position == 0:
            # Long: Williams %R crosses above -80 AND price > 1d EMA34 AND volume confirmation
            long_condition = long_cross_up and curr_close > ema34_1d_aligned[i] and volume_confirm
            
            # Short: Williams %R crosses below -20 AND price < 1d EMA34 AND volume confirmation
            short_condition = short_cross_down and curr_close < ema34_1d_aligned[i] and volume_confirm
            
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

name = "6h_WilliamsR_1dEMA34_Trend_VolumeSpike_v2"
timeframe = "6h"
leverage = 1.0