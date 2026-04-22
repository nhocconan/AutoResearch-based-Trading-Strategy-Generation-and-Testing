#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
    # Works in both bull and bear: breakouts capture directional moves, trend filter avoids counter-trend trades
    # Volume surge confirms breakout strength, reducing false signals
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high of last 20 periods
    upper_band = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lower_band = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period average)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume surge AND daily EMA50 uptrend
            if close[i] > upper_band[i] and vol_surge[i] and close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band with volume surge AND daily EMA50 downtrend
            elif close[i] < lower_band[i] and vol_surge[i] and close[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band
            if position == 1:
                if close[i] < lower_band[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_band[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dEMA50_Trend_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0