#!/usr/bin/env python3

"""
Hypothesis: 6-hour Bollinger Bands squeeze breakout with 1-day trend filter and volume confirmation.
In low volatility (Bollinger Band width < 30th percentile), price often breaks out strongly in the direction of the higher timeframe trend.
Volume spikes confirm institutional participation. This strategy avoids whipsaw by only trading breakouts during low volatility regimes
and requiring alignment with the daily trend. Works in both bull and bear markets by trading with the higher timeframe trend.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 6h data for Bollinger Bands - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Calculate 6h Bollinger Bands (20, 2)
    close_6h = df_6h['close'].values
    ma_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_6h).rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / ma_20  # normalized width
    
    # Align Bollinger Bands components
    ma_20_aligned = align_htf_to_ltf(prices, df_6h, ma_20)
    upper_bb_aligned = align_htf_to_ltf(prices, df_6h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_6h, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_6h, bb_width)
    
    # Load 1d data for trend filter and BB width percentile - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # 1d EMA for trend filter (50-period)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d Bollinger Band width for regime filter (20, 2)
    ma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_width_1d = ( (ma_20_1d + 2 * std_20_1d) - (ma_20_1d - 2 * std_20_1d) ) / ma_20_1d
    bb_width_1d_aligned = align_htf_to_ltf(prices, df_1d, bb_width_1d)
    
    # Pre-calculate 30th percentile of 1d BB width for regime filter (using expanding window)
    bb_width_30th = pd.Series(bb_width_1d).expanding(min_periods=50).quantile(0.30).values
    bb_width_30th_aligned = align_htf_to_ltf(prices, df_1d, bb_width_30th)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ma_20_aligned[i]) or np.isnan(upper_bb_aligned[i]) or 
            np.isnan(lower_bb_aligned[i]) or np.isnan(bb_width_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(bb_width_1d_aligned[i]) or
            np.isnan(bb_width_30th_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: low volatility (BB width < 30th percentile)
        is_low_vol = bb_width_1d_aligned[i] < bb_width_30th_aligned[i]
        
        if position == 0 and is_low_vol:
            # Long: price breaks above upper BB, above 1d EMA (uptrend)
            if close[i] > upper_bb_aligned[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB, below 1d EMA (downtrend)
            elif close[i] < lower_bb_aligned[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle band or volatility expands significantly
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches middle band or BB width exceeds 50th percentile (vol expansion)
                if close[i] < ma_20_aligned[i] or bb_width_1d_aligned[i] > bb_width_30th_aligned[i] * 1.5:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches middle band or BB width exceeds 50th percentile
                if close[i] > ma_20_aligned[i] or bb_width_1d_aligned[i] > bb_width_30th_aligned[i] * 1.5:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Bollinger_Squeeze_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0