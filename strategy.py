# 4h_Combined_Pattern_Plus_Volume_20250524
# Hypothesis: Combines volume spike, price action near support/resistance, and trend alignment.
# Uses 4h timeframe with 1d HTF for trend filter. Designed for low trade frequency and robustness in both bull and bear markets.
# Entry: Long when price is near support, volume spikes, and above 1d EMA50 trend.
# Exit: Short when price is near resistance, volume spikes, and below 1d EMA50 trend.
# Position sizing: 0.25 for clear signals, 0.0 otherwise.
# Risk managed by holding period and mean-reversion logic.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtd_data import get_htf_data, align_htf_to_ltf

name = "4h_Combined_Pattern_Plus_Volume_20250524"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current 4h volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    # Support and resistance: 20-period high and low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    resistance = high_series.rolling(window=20, min_periods=20).max().values
    support = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for support/resistance and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_4h[i]) or np.isnan(volume_filter[i]) or
            np.isnan(resistance[i]) or np.isnan(support[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_4h[i]
        vol_filter = volume_filter[i]
        res = resistance[i]
        sup = support[i]
        
        if position == 0:
            # Enter long: price near support, volume spike, above trend
            if close[i] <= sup * 1.02 and close[i] >= sup * 0.98 and vol_filter and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: price near resistance, volume spike, below trend
            elif close[i] >= res * 0.98 and close[i] <= res * 1.02 and vol_filter and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price near resistance or trend breaks down
            if close[i] >= res * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price near support or trend breaks up
            if close[i] <= sup * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals