#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h Donchian channel direction and volume confirmation
# Long when: BB width < 20th percentile (squeeze) AND price breaks above upper BB AND price > 12h Donchian(20) high AND volume > 2x 20-period average
# Short when: BB width < 20th percentile (squeeze) AND price breaks below lower BB AND price < 12h Donchian(20) low AND volume > 2x 20-period average
# Exit when: price returns to BB middle (20-period SMA) OR BB width expands above 50th percentile (squeeze ends)
# Uses 6h primary timeframe with 12h HTF for trend structure to capture explosive moves after low volatility periods
# Bollinger squeeze identifies coiled price; Donchian breakout confirms directional momentum; volume validates participation
# Discrete sizing (0.25) limits fee drag; squeeze breakouts occur infrequently (target: 12-37 trades/year)
# Works in bull markets (breakouts continue trends) and bear markets (breakdowns accelerate declines)

name = "6h_BB_Squeeze_Donchian_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    donch_high = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    
    # Bollinger Bands on 6h (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # BB width percentiles for squeeze detection (using 50-period lookback)
    bb_width_pct = pd.Series(bb_width).rolling(window=50, min_periods=50).rank(pct=True).values
    squeeze_condition = bb_width_pct < 0.20  # BB width < 20th percentile
    expansion_condition = bb_width_pct > 0.50  # BB width > 50th percentile (exit squeeze)
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(bb_width_pct[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze AND price breaks above upper BB AND price > 12h Donchian high AND volume spike
            if (squeeze_condition[i] and 
                close[i] > upper_bb[i] and 
                close[i] > donch_high_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze AND price breaks below lower BB AND price < 12h Donchian low AND volume spike
            elif (squeeze_condition[i] and 
                  close[i] < lower_bb[i] and 
                  close[i] < donch_low_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to BB middle OR BB expansion (squeeze ends)
            if close[i] < sma_20[i] or expansion_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to BB middle OR BB expansion (squeeze ends)
            if close[i] > sma_20[i] or expansion_condition[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals