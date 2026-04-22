#!/usr/bin/env python3

"""
Hypothesis: 12-hour Donchian channel breakout with daily trend filter and volume confirmation.
Go long when price breaks above 12h Donchian upper band (20-period) and daily trend is up with volume confirmation.
Go short when price breaks below Donchian lower band and daily trend is down with volume confirmation.
Exit on opposite breakout or volatility expansion. Designed for low trade frequency (12-37/year) with trend-following edge.
Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter (EMA 34)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema34 = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    daily_ema34_aligned = align_htf_to_ltf(prices, df_daily, daily_ema34)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(daily_ema34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: breakout above upper band + daily uptrend + volume spike
            if close[i] > donchian_high[i] and daily_ema34_aligned[i] > daily_ema34_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower band + daily downtrend + volume spike
            elif close[i] < donchian_low[i] and daily_ema34_aligned[i] < daily_ema34_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite breakout or volatility expansion (price retracement to midpoint)
            exit_signal = False
            
            # Calculate Donchian midpoint for exit condition
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2
            
            if position == 1:
                # Exit long: breakdown below lower band or price returns to midpoint
                if close[i] < donchian_low[i] or close[i] < donchian_mid:
                    exit_signal = True
            else:  # position == -1
                # Exit short: breakout above upper band or price returns to midpoint
                if close[i] > donchian_high[i] or close[i] > donchian_mid:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0