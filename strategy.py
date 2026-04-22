#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout (20-period) with 1-day EMA50 trend filter and volume confirmation
    # Works in both bull and bear markets: breakouts from established channels capture directional moves
    # EMA50 filters trend direction to avoid counter-trend trades
    # Volume surge confirms breakout strength, reducing false signals
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily EMA50 trend filter
    ema_1d_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume surge AND daily EMA50 uptrend
            if close[i] > donchian_high[i] and vol_surge[i] and close[i] > ema_1d_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band with volume surge AND daily EMA50 downtrend
            elif close[i] < donchian_low[i] and vol_surge[i] and close[i] < ema_1d_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band
            if position == 1:
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dEMA50_Trend_VolumeSurge_v1"
timeframe = "4h"
leverage = 1.0