#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ATR filter and 1d volume confirmation
# Uses discrete position sizing 0.30 to limit trades to ~12-37/year and reduce fee drag
# Works in bull/bear markets: breakout catches trends, ATR filter avoids whipsaws in low volatility, volume confirms institutional participation
# Novelty: Donchian breakout on 6h with multi-timeframe confirmation not recently tested in this session

name = "6h_12h_1d_donchian_breakout_atr_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR(20) for volatility filter
    tr1 = np.abs(high_12h[1:] - low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_12h = wilders_smoothing(tr, 20)
    atr_ma_12h = pd.Series(atr_12h).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 12h and 1d indicators to 6h timeframe
    atr_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_ma_12h)
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(atr_ma_12h_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when current ATR > 20-period MA ATR
        vol_filter = atr_12h[i] > atr_ma_12h_aligned[i] if not np.isnan(atr_12h[i]) else False
        
        # Volume confirmation: current volume > 1.5x average 1d volume
        vol_confirm = volume[i] > 1.5 * avg_volume_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit long if price drops below lowest_20 or volatility/volume conditions fail
            if close[i] < lowest_20[i] or not vol_filter or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short position
            # Exit short if price rises above highest_20 or volatility/volume conditions fail
            if close[i] > highest_20[i] or not vol_filter or not vol_confirm:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Enter long on breakout above highest_20 with vol filter and volume confirmation
            if close[i] > highest_20[i] and vol_filter and vol_confirm:
                position = 1
                signals[i] = 0.30
            # Enter short on breakout below lowest_20 with vol filter and volume confirmation
            elif close[i] < lowest_20[i] and vol_filter and vol_confirm:
                position = -1
                signals[i] = -0.30
    
    return signals