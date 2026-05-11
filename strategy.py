#!/usr/bin/env python3
name = "1d_WeeklyVolatilityBreakout"
timeframe = "1d"
leverage = 1.0

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
    
    # 1d data for weekly volatility calculation (uses previous week's ATR)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly ATR(5) using previous 5 days
    atr_5 = np.zeros(len(close_1d))
    for i in range(5, len(close_1d)):
        tr = np.max([
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        ])
        atr_5[i] = np.mean([
            high_1d[i-4] - low_1d[i-4],
            abs(high_1d[i-4] - close_1d[i-5]),
            abs(low_1d[i-4] - close_1d[i-5]),
            high_1d[i-3] - low_1d[i-3],
            abs(high_1d[i-3] - close_1d[i-4]),
            abs(low_1d[i-3] - close_1d[i-4]),
            high_1d[i-2] - low_1d[i-2],
            abs(high_1d[i-2] - close_1d[i-3]),
            abs(low_1d[i-2] - close_1d[i-3]),
            high_1d[i-1] - low_1d[i-1],
            abs(high_1d[i-1] - close_1d[i-2]),
            abs(low_1d[i-1] - close_1d[i-2]),
            tr
        ])
    
    # Weekly volatility breakout: price breaks above/below previous week's high/low + volatility
    prev_week_high = np.concatenate([[np.nan]*5, high_1d[:-5]]).max(axis=1)  # Max of previous 5 days
    prev_week_low = np.concatenate([[np.nan]*5, low_1d[:-5]]).min(axis=1)    # Min of previous 5 days
    
    # Add volatility buffer
    upper_band = prev_week_high + (atr_5 * 0.5)
    lower_band = prev_week_low - (atr_5 * 0.5)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # Align weekly bands to daily timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 5)  # Ensure enough data for volatility and volume
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above upper band with volume confirmation
            if close[i] > upper_band_aligned[i] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with volume confirmation
            elif close[i] < lower_band_aligned[i] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price closes below lower band or volatility drops
            if close[i] < lower_band_aligned[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price closes above upper band or volatility drops
            if close[i] > upper_band_aligned[i] or not vol_confirm[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals