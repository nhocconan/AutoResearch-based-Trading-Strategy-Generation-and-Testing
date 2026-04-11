#!/usr/bin/env python3
"""
1d_1w_donchian_volume_trend_v1
Strategy: 1d Donchian breakout with volume confirmation and weekly trend filter
Timeframe: 1d
Leverage: 1.0
Hypothesis: Buy when price breaks above 20-day Donchian high with above-average volume in a bullish weekly trend (price above weekly EMA50). Sell/short when price breaks below 20-day Donchian low with above-average volume in a bearish weekly trend (price below weekly EMA50). Designed to capture strong trends with volume confirmation while avoiding false breakouts in chop. Works in both bull/bear markets by following the weekly trend direction. Target: 20-60 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1d Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        uptrend_1w = close[i] > ema_50_1w_aligned[i]
        downtrend_1w = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_above_avg = volume[i] > vol_ma[i]
        
        # Breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous period's low
        
        # Long: bullish breakout with volume in uptrend
        long_signal = breakout_up and vol_above_avg and uptrend_1w
        
        # Short: bearish breakout with volume in downtrend
        short_signal = breakout_down and vol_above_avg and downtrend_1w
        
        # Exit when price returns to the opposite Donchian level
        exit_long = position == 1 and close[i] < donchian_low[i]
        exit_short = position == -1 and close[i] > donchian_high[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Buy when price breaks above 20-day Donchian high with above-average volume in a bullish weekly trend (price above weekly EMA50). Sell/short when price breaks below 20-day Donchian low with above-average volume in a bearish weekly trend (price below weekly EMA50). Designed to capture strong trends with volume confirmation while avoiding false breakouts in chop. Works in both bull/bear markets by following the weekly trend direction. Target: 20-60 total trades over 4 years.