#!/usr/bin/env python3
"""
12h_Rolling_Regression_Trend_v1
Hypothesis: Use 12h linear regression slope over 20 periods as primary trend filter, combined with 1-day volume spike and price position relative to VWAP for entry. Long when: regression slope > 0, price > VWAP, and volume spike. Short when: slope < 0, price < VWAP, and volume spike. Exits when slope changes sign. Designed for 12h timeframe to capture medium-term trends with low turnover (target: 15-35 trades/year). Works in bull markets via long positions in uptrends and bear markets via short positions in downtrends. Volume spike ensures institutional participation, reducing false breakouts.
"""

name = "12h_Rolling_Regression_Trend_v1"
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
    
    # === 1D Data for Volume Spike and VWAP ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1D VWAP calculation (typical price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_1d = (typical_price_1d * volume_1d).cumsum() / volume_1d.cumsum()
    
    # 1D Volume spike: current volume > 2x 20-period average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_1d * 2.0)
    
    # Align 1D indicators to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # === 12h Linear Regression Slope (20-period) ===
    def rolling_regression_slope(arr, window):
        n = len(arr)
        slope = np.full(n, np.nan)
        if n < window:
            return slope
        for i in range(window - 1, n):
            y = arr[i - window + 1:i + 1]
            x = np.arange(window)
            if np.all(np.isnan(y)) or np.all(np.isnan(x)):
                slope[i] = np.nan
            else:
                # Remove NaNs if any
                valid = ~(np.isnan(y) | np.isnan(x))
                if np.sum(valid) < 2:
                    slope[i] = np.nan
                else:
                    x_valid = x[valid]
                    y_valid = y[valid]
                    slope[i] = np.polyfit(x_valid, y_valid, 1)[0]
        return slope
    
    # Calculate 20-period rolling regression slope on 12h close
    reg_slope = rolling_regression_slope(close, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 bars for regression + 1d data alignment)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(reg_slope[i]) or 
            np.isnan(vwap_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: positive regression slope, price above VWAP, volume spike
            if reg_slope[i] > 0 and close[i] > vwap_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: negative regression slope, price below VWAP, volume spike
            elif reg_slope[i] < 0 and close[i] < vwap_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: regression slope turns negative
            if reg_slope[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: regression slope turns positive
            if reg_slope[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals