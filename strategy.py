#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w HMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND 1w HMA(21) is rising AND volume > 1.5x 20-period average volume
# Short when price breaks below Donchian(20) low AND 1w HMA(21) is falling AND volume > 1.5x 20-period average volume
# Exit when price crosses Donchian(20) midpoint or volume drops below average
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown
# Designed for 7-25 trades/year on 1d timeframe with strong trend filters to avoid overtrading

name = "1d_Donchian20_1wHMA_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on 1w close for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMA for half length
    wma_half = np.array([wma(close_1w[i:i+half_len], half_len) 
                         if i+half_len <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    
    # Calculate WMA for full length
    wma_full = np.array([wma(close_1w[i:i+21], 21) 
                         if i+21 <= len(close_1w) else np.nan 
                         for i in range(len(close_1w))])
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw_hma with sqrt_len
    hma_21_1w = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) 
                          if i+sqrt_len <= len(raw_hma) else np.nan 
                          for i in range(len(raw_hma))])
    
    # Align HTF HMA to LTF (1d) - wait for completed 1w bar
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # Calculate Donchian channels (20-period) on 1d
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(vol_avg[i]) or np.isnan(hma_21_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high AND 1w HMA is rising AND volume > 1.5x average
            if close[i] > donchian_high[i] and hma_21_1w_aligned[i] > hma_21_1w_aligned[i-1] and volume[i] > 1.5 * vol_avg[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low AND 1w HMA is falling AND volume > 1.5x average
            elif close[i] < donchian_low[i] and hma_21_1w_aligned[i] < hma_21_1w_aligned[i-1] and volume[i] > 1.5 * vol_avg[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian mid OR volume drops below average
            if close[i] < donchian_mid[i] or volume[i] < vol_avg[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian mid OR volume drops below average
            if close[i] > donchian_mid[i] or volume[i] < vol_avg[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals