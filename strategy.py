#!/usr/bin/env python3
name = "1d_RVOL_Breakout_1wTrend"
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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Weekly EMA(50) for long-term trend
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily 20-period moving average and standard deviation for RVOL
    close_series = pd.Series(close)
    volume_series = pd.Series(volume)
    
    ma20 = close_series.rolling(window=20, min_periods=20).mean().values
    std20 = close_series.rolling(window=20, min_periods=20).std().values
    
    # 20-day average volume
    avg_vol20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Bollinger Bands (20, 2)
    upper_band = ma20 + 2 * std20
    lower_band = ma20 - 2 * std20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ma20[i]) or 
            np.isnan(std20[i]) or np.isnan(avg_vol20[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below weekly EMA50
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Relative volume (current volume / 20-day average volume)
        rvol = volume[i] / avg_vol20[i] if avg_vol20[i] > 0 else 0
        vol_surge = rvol > 2.0  # Volume at least 2x average
        
        # Bollinger Band breakout
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        if position == 0:
            # LONG: Weekly uptrend + volume surge + breakout above upper band
            if weekly_uptrend and vol_surge and breakout_up:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + volume surge + breakout below lower band
            elif weekly_downtrend and vol_surge and breakout_down:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend weakens or price returns below middle band
            if not weekly_uptrend or close[i] < ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend weakens or price returns above middle band
            if not weekly_downtrend or close[i] > ma20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals