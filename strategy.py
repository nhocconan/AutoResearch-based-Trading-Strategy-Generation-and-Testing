#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend trades during ranges
# Volume confirmation (>1.5x 20-period average) filters low-quality breakouts
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_Donchian20_Volume_1wEMA50_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20, 50)  # warmup for Donchian, volume MA, 1w EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian high with close above weekly EMA50
                if curr_high > donchian_high[i-1] and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian low with close below weekly EMA50
                elif curr_low < donchian_low[i-1] and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price closes below Donchian low (trend reversal)
            if curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian high (trend reversal)
            if curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals