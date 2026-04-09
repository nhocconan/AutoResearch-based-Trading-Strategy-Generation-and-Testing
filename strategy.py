#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d ATR filter
# Long when price breaks above 20-period high + volume > 1.5x 20-period avg volume + ATR(14) < 0.03*price (low volatility)
# Short when price breaks below 20-period low + volume > 1.5x 20-period avg volume + ATR(14) < 0.03*price (low volatility)
# Uses discrete position sizing 0.25 to limit trades to ~20-50/year and reduce fee drag
# Works in bull/bear markets: breakouts capture momentum in both directions with volatility filter to avoid chop

name = "4h_1d_donchian_breakout_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14)
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
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
    
    atr_1d = wilders_smoothing(tr, 14)
    atr_ratio_1d = np.where(close_1d > 0, atr_1d / close_1d, np.nan)
    
    # Calculate 1d 20-period average volume
    vol_s_1d = pd.Series(volume_1d)
    avg_vol_20_1d = vol_s_1d.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    donchian_high = high_s.rolling(window=20, min_periods=20).max().values
    donchian_low = low_s.rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_ratio_1d_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i]) or
            np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / avg_vol_20_1d_aligned[i] if avg_vol_20_1d_aligned[i] > 0 else 0
        low_volatility = atr_ratio_1d_aligned[i] < 0.03
        high_volume = vol_ratio > 1.5
        
        if position == 1:  # Long position
            # Exit long if price breaks below Donchian low or volatility increases
            if close[i] <= donchian_low[i] or not low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit short if price breaks above Donchian high or volatility increases
            if close[i] >= donchian_high[i] or not low_volatility:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long on Donchian high breakout with volume and low volatility confirmation
            if close[i] > donchian_high[i] and high_volume and low_volatility:
                position = 1
                signals[i] = 0.25
            # Enter short on Donchian low breakdown with volume and low volatility confirmation
            elif close[i] < donchian_low[i] and high_volume and low_volatility:
                position = -1
                signals[i] = -0.25
    
    return signals