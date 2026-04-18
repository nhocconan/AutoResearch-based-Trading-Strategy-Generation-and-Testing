#!/usr/bin/env python3
"""
12h Donchian(20) Breakout + Daily Trend Filter + Volume Spike
Breakout long when price breaks above 20-period high + daily EMA34 up + volume spike
Breakout short when price breaks below 20-period low + daily EMA34 down + volume spike
Exit when price crosses back through 20-period opposite band
Uses 12h price action with 1d trend filter to capture multi-day trends
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian calculations
    
    for i in range(start_idx, n):
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_band = period20_high[i]
        lower_band = period20_low[i]
        ema_trend = ema34_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band + daily uptrend + volume spike
            if price > upper_band and price > ema_trend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + daily downtrend + volume spike
            elif price < lower_band and price < ema_trend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower band (contrarian exit)
            if price < lower_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper band (contrarian exit)
            if price > upper_band:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0