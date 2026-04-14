#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 12-hour trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band (20), 12h EMA > 12h EMA 50 periods ago (uptrend),
# and volume > 1.5x 4h volume average (20). Short when price breaks below 4h Donchian lower band
# with 12h EMA < 12h EMA 50 periods ago (downtrend) and volume spike.
# Exit when price crosses back below/above 4h Donchian midline.
# Designed to capture strong trends with volume confirmation, avoiding choppy markets.
# Target: 25-60 trades per symbol over 4 years (6-15/year).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 12h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA (50-period) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 60  # for Donchian and EMA calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = vol_4h[i] if i < len(vol_4h) else vol_4h[-1]
        
        if position == 0:
            # Long setup: break above Donchian high with volume spike and 12h uptrend
            if (price > donchian_high_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                ema_12h_aligned[i] > ema_12h_aligned[i-50]):    # 12h uptrend (current > 50 periods ago)
                position = 1
                signals[i] = position_size
            # Short setup: break below Donchian low with volume spike and 12h downtrend
            elif (price < donchian_low_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                  ema_12h_aligned[i] < ema_12h_aligned[i-50]):    # 12h downtrend (current < 50 periods ago)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian mid
            if price < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian mid
            if price > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0