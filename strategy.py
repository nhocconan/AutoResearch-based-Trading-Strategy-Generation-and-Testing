#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + 1w EMA21 Trend + Volume Spike Filter
# - Long when close > 20-day high + close > weekly EMA21 + volume > 1.5x 20-day avg volume
# - Short when close < 20-day low + close < weekly EMA21 + volume > 1.5x 20-day avg volume
# - Exit on opposite breakout or trend reversal
# - Uses weekly EMA21 to filter for higher timeframe trend direction
# - Volume spike confirms breakout strength
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Load 1w data for EMA21 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-day Donchian channels on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume on 1d
    avg_volume = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate EMA21 on 1w
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA21 to 1d timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Skip if NaN in indicators
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(avg_volume[i]) or np.isnan(ema21_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        vol_threshold = 1.5 * avg_volume[i]
        ema_trend = ema21_1w_aligned[i]
        
        if position == 0:
            # Long entry: breakout above Donchian high + above weekly EMA21 + volume spike
            if price > donchian_high[i] and price > ema_trend and vol > vol_threshold:
                signals[i] = 0.25
                position = 1
            # Short entry: breakout below Donchian low + below weekly EMA21 + volume spike
            elif price < donchian_low[i] and price < ema_trend and vol > vol_threshold:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: breakdown below Donchian low or trend reversal
            if price < donchian_low[i] or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: breakout above Donchian high or trend reversal
            if price > donchian_high[i] or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA21_VolumeFilter"
timeframe = "1d"
leverage = 1.0