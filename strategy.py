#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike (2.0x 30-period avg) and 12h EMA50 trend filter
# Donchian channels capture price structure; breakouts signal momentum with volume confirmation
# 12h EMA50 ensures medium-term trend alignment to avoid counter-trend trades
# Discrete sizing (0.25) minimizes fee churn; target 75-200 total trades over 4 years
# Works in bull/bear: breakouts catch strong moves, volume filter ensures legitimacy, EMA50 trend filter avoids whipsaws

name = "4h_Donchian20_Breakout_VolumeSpike_12hEMA50_Trend_v1"
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
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30, 14, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        
        if position == 0:  # Flat - look for new entries
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper Donchian + above 12h EMA50
                if curr_high > curr_highest_20 and curr_close > curr_ema_50_12h:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below lower Donchian + below 12h EMA50
                elif curr_low < curr_lowest_20 and curr_close < curr_ema_50_12h:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below lower Donchian
            if curr_low < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian
            if curr_high > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals