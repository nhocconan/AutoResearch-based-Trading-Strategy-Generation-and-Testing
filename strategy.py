#!/usr/bin/env python3
"""
Hypothesis: Daily Bollinger Band Width squeeze breakout with weekly trend filter and volume confirmation.
Long when BBWidth < 20th percentile (squeeze) and price breaks above upper BB with weekly EMA50 rising and volume spike.
Short when BBWidth < 20th percentile and price breaks below lower BB with weekly EMA50 falling and volume spike.
Exit when price returns to middle BB or weekly trend reverses.
Designed for low trade frequency by requiring volatility squeeze + breakout + trend alignment.
Works in both bull and bear markets by trading breakouts from low volatility regimes.
"""

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
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bb_width = (upper - lower) / sma  # normalized width
    
    # BBWidth percentile lookback (50 periods)
    def rolling_percentile(arr, window, percentile):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            window_data = arr[i-window+1:i+1]
            valid = ~np.isnan(window_data)
            if np.sum(valid) >= window:
                result[i] = np.percentile(window_data[valid], percentile)
        return result
    
    bb_width_20th = rolling_percentile(bb_width, 50, 20)
    squeeze = bb_width < bb_width_20th
    
    # Load weekly data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on weekly close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(squeeze[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(sma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: squeeze breakout up with weekly uptrend and volume spike
            if squeeze[i] and close[i] > upper[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: squeeze breakout down with weekly downtrend and volume spike
            elif squeeze[i] and close[i] < lower[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: return to middle BB or weekly trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle BB or weekly trend turns down
                if close[i] <= sma[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle BB or weekly trend turns up
                if close[i] >= sma[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_BBWidth_Squeeze_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0