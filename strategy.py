# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_LarryWilliamsVolatilityBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    Larry Williams Volatility Breakout (LVB) with 1d trend filter and volume confirmation.
    Long: Open > previous close + k * (previous high - previous low) with volume > 1.5x avg and price > 1d EMA(50)
    Short: Open < previous close - k * (previous high - previous low) with volume > 1.5x avg and price < 1d EMA(50)
    Exit: Close crosses the previous close (mean reversion)
    k = 0.55 (optimized for 6h BTC/ETH)
    Target: 15-35 trades/year on 6h timeframe
    """
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate previous bar's range for LVB
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_range = prev_high - prev_low
    
    # LVB levels: k = 0.55
    k = 0.55
    long_trigger = prev_close + k * prev_range
    short_trigger = prev_close - k * prev_range
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(long_trigger[i]) or np.isnan(short_trigger[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Open breaks above LVB trigger with volume confirmation and above 1d EMA trend
            if open_price[i] > long_trigger[i] and vol_ok and open_price[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Open breaks below LVB trigger with volume confirmation and below 1d EMA trend
            elif open_price[i] < short_trigger[i] and vol_ok and open_price[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses below previous close (mean reversion)
            if close[i] < prev_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses above previous close (mean reversion)
            if close[i] > prev_close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals