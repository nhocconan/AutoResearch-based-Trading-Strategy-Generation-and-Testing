# [Hypothesis] 6h timeframe with 12h/1d multi-timeframe confirmation: 
# Strategy uses 12h Donchian breakout for trend direction and 1d volume confirmation for momentum.
# Works in bull/bear markets by trading with higher timeframe trend. Volume filter reduces whipsaw.
# Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_DonchianBreakout_12hTrend_1dVolConfirm"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend (Donchian channel)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Vectorized rolling max/min
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_ma_today = vol_ma_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = vol_current > 1.5 * vol_ma_today
        
        if position == 0:
            # Long entry: price breaks above 12h Donchian high with volume confirmation
            if price > donchian_high_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below 12h Donchian low with volume confirmation
            elif price < donchian_low_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 12h Donchian low
            if price < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 12h Donchian high
            if price > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals