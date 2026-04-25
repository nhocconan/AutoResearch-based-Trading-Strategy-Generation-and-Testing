#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter (price >/< EMA34) and volume confirmation (>1.5x 20-bar avg). Enters long when price breaks above R1 in 1d uptrend, short when breaks below S1 in 1d downtrend. Uses discrete sizing (0.25) to limit fee churn. Designed for 4h timeframe with ~20-50 trades/year, works in bull/bear by following 1d trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 4h using previous day's OHLC
    # Camarilla levels are calculated from previous day's range
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    # We need previous day's OHLC, so we'll use 1d data shifted by 1
    
    # Get previous day's OHLC (1d data shifted)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_open_1d = np.roll(df_1d['open'].values, 1)
    # Set first value to NaN since no previous day
    prev_close_1d[0] = np.nan
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_open_1d[0] = np.nan
    
    # Align previous day's OHLC to 4h timeframe
    prev_close_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    prev_high_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Calculate Camarilla R1 and S1 levels
    # R1 = prev_close + 1.1*(prev_high - prev_low)/12
    # S1 = prev_close - 1.1*(prev_high - prev_low)/12
    camarilla_range = prev_high_1d_aligned - prev_low_1d_aligned
    r1_level = prev_close_1d_aligned + 1.1 * camarilla_range / 12
    s1_level = prev_close_1d_aligned - 1.1 * camarilla_range / 12
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough data for EMA and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_level[i]) or 
            np.isnan(s1_level[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 in 1d uptrend with volume confirmation
            bullish_setup = (close[i] > r1_level[i]) and (close_1d[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: price breaks below S1 in 1d downtrend with volume confirmation
            bearish_setup = (close[i] < s1_level[i]) and (close_1d[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if bullish_setup:
                signals[i] = 0.25
                position = 1
            elif bearish_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below S1 OR trend turns down
            if (close[i] < s1_level[i]) or (close_1d[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above R1 OR trend turns up
            if (close[i] > r1_level[i]) or (close_1d[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0