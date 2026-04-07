#!/usr/bin/env python3
"""
12h_donchian_breakout_1d_trend_volume_v1
Hypothesis: On 12h timeframe, buy when price breaks above 20-period Donchian high with 1d uptrend (EMA50 > EMA200) and volume confirmation; sell when price breaks below 20-period Donchian low with 1d downtrend (EMA50 < EMA200) and volume confirmation. Exit on opposite Donchian breakout or trend reversal. This strategy captures medium-term trends with institutional volume confirmation, working in both bull and bear markets by following the 1d trend. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d trend data (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMAs on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    
    # Align to 12h timeframe (with shift(1) for completed bars only)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d trend: True if EMA50 > EMA200 (uptrend), False if EMA50 < EMA200 (downtrend)
    trend_up = ema50_1d_aligned > ema200_1d_aligned
    trend_down = ema50_1d_aligned < ema200_1d_aligned
    
    # Donchian channels (20-period)
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(period, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price breaks below Donchian low
            if close[i] < lowest_low[i]:
                exit_long = True
            # Exit if 1d trend turns down
            elif trend_down[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price breaks above Donchian high
            if close[i] > highest_high[i]:
                exit_short = True
            # Exit if 1d trend turns up
            elif trend_up[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above Donchian high, 1d uptrend, and volume confirmation
            if close[i] > highest_high[i] and trend_up[i] and vol_confirm:
                long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below Donchian low, 1d downtrend, and volume confirmation
            if close[i] < lowest_low[i] and trend_down[i] and vol_confirm:
                short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals