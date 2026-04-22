#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1w trend filter and volume confirmation
    # Works in both bull and bear markets: breakouts from price channels capture directional moves
    # Weekly trend filter ensures alignment with long-term momentum
    # Volume surge confirms breakout strength
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA40 trend filter
    ema_1w_40 = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema_1w_40_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_40)
    
    # 12h Donchian Channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian upper and lower bands
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_40_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume surge AND weekly EMA40 uptrend
            if close[i] > donch_high[i] and vol_surge[i] and close[i] > ema_1w_40_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower band with volume surge AND weekly EMA40 downtrend
            elif close[i] < donch_low[i] and vol_surge[i] and close[i] < ema_1w_40_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to Donchian midpoint or opposite band touch
            midpoint = (donch_high[i] + donch_low[i]) / 2
            if position == 1:
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wEMA40_Trend_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0