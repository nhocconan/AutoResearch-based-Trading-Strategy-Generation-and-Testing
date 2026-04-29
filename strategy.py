#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume spike and 1w EMA50 trend filter
# Uses proven Donchian breakout structure with weekly EMA50 trend filter
# Volume spike (>2.0x 20-period average) confirms institutional participation
# 1d timeframe minimizes fee drag while capturing multi-day trends
# Works in bull/bear: volume confirms breakout validity, 1w EMA50 filters counter-trend noise
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# Novelty: Applying Donchian+Volume+Trend framework to 1d with 1w HTF filter

name = "1d_Donchian20_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # 1w EMA50 for trend filter (HTF = 1w as requested)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_high_20 = high_20[i]
        curr_low_20 = low_20[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper Donchian with volume and above 1w EMA50
                if curr_high > curr_high_20 and curr_close > curr_ema_50:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below lower Donchian with volume and below 1w EMA50
                elif curr_low < curr_low_20 and curr_close < curr_ema_50:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below lower Donchian (reversal signal)
            if curr_low < curr_low_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian (reversal signal)
            if curr_high > curr_high_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals