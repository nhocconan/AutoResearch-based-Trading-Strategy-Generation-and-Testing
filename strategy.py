#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1w EMA Trend Filter and 1d Volume Spike
# Takes long when price breaks above 12h Donchian upper band with weekly EMA uptrend and daily volume spike
# Takes short when price breaks below 12h Donchian lower band with weekly EMA downtrend and daily volume spike
# Exits when price returns to the 12h Donchian midline
# Designed to capture strong trends with volume confirmation, avoiding false breakouts in low volume
# Target: 20-40 trades per symbol over 4 years (5-10/year) to stay within fee limits

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h, 1d, and 1w data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1w EMA (50-period) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Long setup: break above Donchian high with weekly uptrend and volume spike
            if (price > donchian_high_aligned[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[max(0, i-1)] and  # Weekly EMA rising
                vol_1d_current > 2.0 * vol_ma_1d_aligned[i]):             # Strong volume spike
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with weekly downtrend and volume spike
            elif (price < donchian_low_aligned[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[max(0, i-1)] and  # Weekly EMA falling
                  vol_1d_current > 2.0 * vol_ma_1d_aligned[i]):              # Strong volume spike
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midline
            if price <= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midline
            if price >= donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Donchian_Breakout_1wEMA_1dVolume"
timeframe = "12h"
leverage = 1.0