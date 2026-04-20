# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) Breakout + 1d Volume Spike + 1d EMA Trend Filter
# - Long when price breaks above Donchian high(20) + volume > 2x 20-period avg + 1d EMA50 up
# - Short when price breaks below Donchian low(20) + volume > 2x 20-period avg + 1d EMA50 down
# - Exit when price crosses back through Donchian midline (20-period avg) or trend reverses
# - Uses Donchian for breakout structure, volume for confirmation, 1d EMA for trend filter
# - Designed for 4h timeframe with selective entries to avoid overtrading
# - Target: 15-40 trades per year per symbol (60-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 4h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high/low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2  # Midline for exit
    
    # Volume average (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + bullish 1d trend
            if price > donch_high[i] and vol > 2.0 * vol_ma[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + bearish 1d trend
            elif price < donch_low[i] and vol > 2.0 * vol_ma[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian midline or bearish 1d trend
            if price < donch_mid[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian midline or bullish 1d trend
            if price > donch_mid[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_20_1dVolumeSpike_EMA50Trend"
timeframe = "4h"
leverage = 1.0