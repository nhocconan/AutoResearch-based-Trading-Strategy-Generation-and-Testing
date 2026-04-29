#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with ATR-based volume spike and 1d EMA50 trend filter
# Donchian breakouts capture structural moves; volume >1.5x ATR-scaled 20-bar MA confirms strength
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Target: 75-150 total trades over 4 years (19-37/year) for 4h timeframe
# Works in bull/bear: volume filter reduces false breakouts, daily trend filter improves win rate

name = "4h_Donchian20_VolumeSpike_ATR_1dEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for volatility-based volume threshold
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x (20-period volume MA * ATR ratio to normalize)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_atr_ratio = atr_14 / pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_threshold = 1.5 * vol_ma_20 * (1 + vol_atr_ratio)  # dynamic threshold based on volatility
    volume_confirm = volume > vol_threshold
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 50)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(volume_confirm[i]) or np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_50 = ema_50_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish entry: price breaks above upper Donchian with volume and above 1d EMA50
                if curr_high > curr_highest_20 and curr_close > curr_ema_50:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below lower Donchian with volume and below 1d EMA50
                elif curr_low < curr_lowest_20 and curr_close < curr_ema_50:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when price breaks below lower Donchian (reversal signal)
            if curr_low < curr_lowest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian (reversal signal)
            if curr_high > curr_highest_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals