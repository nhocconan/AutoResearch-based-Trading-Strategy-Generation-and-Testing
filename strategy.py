#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Williams Fractal reversals with volume confirmation
# - Bearish fractal (sell signal): 5-bar pattern where middle bar is highest high
# - Bullish fractal (buy signal): 5-bar pattern where middle bar is lowest low
# - Enter on fractal break in direction of trend with volume confirmation
# - Uses 1-week EMA50 as trend filter to avoid counter-trend trades
# - Designed for low-frequency, high-conviction trades in both bull and bear markets
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_WilliamsFractal_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Fractals
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Williams Fractals on 1d data
    # Bearish fractal: high[n-2] is highest of [n-4, n-3, n-2, n-1, n]
    # Bullish fractal: low[n-2] is lowest of [n-4, n-3, n-2, n-1, n]
    high_arr = df_1d['high'].values
    low_arr = df_1d['low'].values
    
    bearish_fractal = np.zeros(len(high_arr))
    bullish_fractal = np.zeros(len(low_arr))
    
    for i in range(2, len(high_arr) - 2):
        if (high_arr[i] >= high_arr[i-2] and high_arr[i] >= high_arr[i-1] and 
            high_arr[i] >= high_arr[i+1] and high_arr[i] >= high_arr[i+2]):
            bearish_fractal[i] = high_arr[i]
        if (low_arr[i] <= low_arr[i-2] and low_arr[i] <= low_arr[i-1] and 
            low_arr[i] <= low_arr[i+1] and low_arr[i] <= low_arr[i+2]):
            bullish_fractal[i] = low_arr[i]
    
    # Williams fractals need 2 extra 1d bars for confirmation
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Require strong volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(bearish_fractal_aligned[i]) or np.isnan(bullish_fractal_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above bullish fractal with volume
            # Only in uptrend (price above weekly EMA50)
            if (bullish_fractal_aligned[i] > 0 and 
                close[i] > bullish_fractal_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below bearish fractal with volume
            # Only in downtrend (price below weekly EMA50)
            elif (bearish_fractal_aligned[i] > 0 and 
                  close[i] < bearish_fractal_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below bullish fractal (invalidates signal)
            if bullish_fractal_aligned[i] > 0 and close[i] < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above bearish fractal (invalidates signal)
            if bearish_fractal_aligned[i] > 0 and close[i] > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals