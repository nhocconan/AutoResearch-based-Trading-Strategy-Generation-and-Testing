#!/usr/bin/env python3
# Hypothesis: 12h Bollinger Band squeeze with 1d trend filter and volume confirmation
# Long when price breaks above upper BB, 1d EMA50 trending up, and volume > 2x 20-period average
# Short when price breaks below lower BB, 1d EMA50 trending down, and volume > 2x 20-period average
# Exit when price returns to middle BB or trend reverses
# Bollinger Band squeeze identifies low volatility periods preceding breakouts
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Position size: 0.25 to limit drawdown during volatile markets

name = "12h_BB_Squeeze_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = bb_mid + (bb_std_dev * bb_std)
    bb_lower = bb_mid - (bb_std_dev * bb_std)
    bb_mid_vals = bb_mid.values
    bb_upper_vals = bb_upper.values
    bb_lower_vals = bb_lower.values
    
    # Bollinger Band width for squeeze detection
    bb_width = (bb_upper_vals - bb_lower_vals) / bb_mid_vals
    bb_width_ma = pd.Series(bb_width).rolling(window=50, min_periods=50).mean()
    squeeze_condition = bb_width < (0.5 * bb_width_ma.values)  # Width less than 50% of its MA
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 slope for trend direction (rising/falling)
    ema50_slope = np.diff(ema50_1d_aligned, prepend=ema50_1d_aligned[0])
    ema50_rising = ema50_slope > 0
    ema50_falling = ema50_slope < 0
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for BB and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_mid_vals[i]) or np.isnan(bb_upper_vals[i]) or 
            np.isnan(bb_lower_vals[i]) or np.isnan(squeeze_condition[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or 
            np.isnan(ema50_falling[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: BB squeeze breakout up, 1d EMA50 rising, volume spike
            if (close[i] > bb_upper_vals[i] and 
                squeeze_condition[i] and 
                ema50_rising[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: BB squeeze breakout down, 1d EMA50 falling, volume spike
            elif (close[i] < bb_lower_vals[i] and 
                  squeeze_condition[i] and 
                  ema50_falling[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle BB or trend turns bearish
            if (close[i] < bb_mid_vals[i]) or (not ema50_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle BB or trend turns bullish
            if (close[i] > bb_mid_vals[i]) or (not ema50_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals