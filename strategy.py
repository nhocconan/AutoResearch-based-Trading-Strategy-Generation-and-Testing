# 4h_Donchian20_Breakout_Trend_Volume - 4h breakout with 1d trend and volume confirmation
# Uses Donchian(20) breakout as primary signal, 1d EMA50 for trend filter, volume spike for confirmation
# Long when price breaks above Donchian upper + above 1d EMA50 + volume > 1.5x 20-period average
# Short when price breaks below Donchian lower + below 1d EMA50 + volume > 1.5x 20-period average
# Exit when price returns to Donchian middle or trend contradicts
# Position size: 0.28 (28% of capital) balances return and drawdown
# Designed to work in trending markets via EMA filter and avoid whipsaws with volume confirmation

name = "4h_Donchian20_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_max + low_min) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper + above 1d EMA50 + volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema50_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price breaks below Donchian lower + below 1d EMA50 + volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema50_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle OR trend turns bearish
            if (close[i] < donchian_middle[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price returns to Donchian middle OR trend turns bullish
            if (close[i] > donchian_middle[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals