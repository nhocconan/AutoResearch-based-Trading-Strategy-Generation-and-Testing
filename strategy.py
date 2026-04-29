#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves; 1w EMA50 ensures alignment with weekly trend
# Volume spike >2.0x confirms participation; discrete sizing (0.25) minimizes fee churn
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via trend filter.

name = "12h_Donchian20_VolumeSpike_1wEMA50_Trend_v1"
timeframe = "12h"
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
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30, 14, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above Donchian upper + above 1w EMA50
                if curr_high > highest_20[i] and curr_close > curr_ema_50_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below Donchian lower + below 1w EMA50
                elif curr_low < lowest_20[i] and curr_close < curr_ema_50_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower
            if curr_low < lowest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper
            if curr_high > highest_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals