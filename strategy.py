#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1-day volume confirmation and 1-week trend filter.
# Long when: price breaks above Donchian(20) high, volume > 1.5x 20-period average, 1w EMA(50) rising
# Short when: price breaks below Donchian(20) low, volume > 1.5x 20-period average, 1w EMA(50) falling
# Exit when price crosses back through Donchian(20) midline (10-period average).
# Donchian channels provide clear breakout levels; volume confirms conviction; 1w EMA filters counter-trend moves.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_Volume_1wEMA"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    # 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Donchian(20) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 20-period volume average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    # Rising/falling EMA: current > previous
    ema_rising = np.where(ema_50_1w_aligned > np.roll(ema_50_1w_aligned, 1), 1, -1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_dir = ema_rising[i]
        
        if position == 0:
            # Long entry: breakout above Donchian high, volume confirmation, 1w EMA rising
            if price > high_20[i] and vol > 1.5 * vol_ma and ema_dir == 1:
                signals[i] = 0.25
                position = 1
            # Short entry: breakdown below Donchian low, volume confirmation, 1w EMA falling
            elif price < low_20[i] and vol > 1.5 * vol_ma and ema_dir == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midline
            if price < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midline
            if price > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals