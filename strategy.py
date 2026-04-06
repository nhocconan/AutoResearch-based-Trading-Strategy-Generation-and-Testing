#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation
# Long when price breaks above Donchian upper band AND price > EMA50 AND volume > 2x average
# Short when price breaks below Donchian lower band AND price < EMA50 AND volume > 2x average
# Exit when price crosses back through Donchian midline (10-period) OR volatility filter triggers
# Uses weekly timeframe for trend filter to reduce trades and improve trend accuracy
# Works in both bull and bear by only taking trades in direction of higher timeframe trend

name = "1d_donchian_1w_ema_vol_v4"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on daily data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Weekly EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(ema50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or close[i] < ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or close[i] > ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of weekly trend with volume confirmation
            # Long: price breaks above Donchian upper AND above weekly EMA50 + volume spike
            if (close[i] > donchian_upper[i] and close[i] > ema50_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND below weekly EMA50 + volume spike
            elif (close[i] < donchian_lower[i] and close[i] < ema50_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals