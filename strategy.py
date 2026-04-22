#!/usr/bin/env python3

"""
Hypothesis: 4-hour Bollinger Band Width squeeze breakout with 1-day EMA trend filter and volume confirmation.
Trades breakouts after low volatility contractions (squeeze) in the direction of the daily EMA trend.
Uses volume spike to confirm institutional interest. Designed for low trade frequency (20-50 trades/year)
to minimize fee drift and work in both bull and bear markets by combining volatility contraction
with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for trend filter and Bollinger calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Bollinger Bands (20, 2.0) for squeeze detection
    close_1d_series = pd.Series(close_1d)
    bb_middle = close_1d_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Bollinger Band Width squeeze: current width < 50th percentile of past 50 days
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=50).quantile(0.50).values
    squeeze = bb_width < bb_width_percentile
    
    # Align daily indicators to 4h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(squeeze_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike and squeeze_aligned[i]:
            # Long: price breaks above upper Bollinger Band with uptrend bias
            if close[i] > bb_upper[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Bollinger Band with downtrend bias
            elif close[i] < bb_lower[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to middle Bollinger Band or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below middle band or closes below daily EMA
                if close[i] < bb_middle[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above middle band or closes above daily EMA
                if close[i] > bb_middle[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Bollinger_Width_Squeeze_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0