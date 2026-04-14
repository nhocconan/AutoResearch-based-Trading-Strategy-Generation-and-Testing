#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 1d ADX Trend Filter and Volume Spike
# Uses Donchian Channel (20) for breakout entries in trending markets
# 1d ADX (>25) ensures we only trade when strong trend is present
# Volume confirmation (>1.5x average) filters for institutional participation
# Designed to capture trend continuation moves while avoiding choppy markets
# Target: 25-40 trades/year (100-160 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d ADX data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX (14) on daily data
    plus_dm = np.diff(df_1d['high'], prepend=df_1d['high'].iloc[0])
    minus_dm = np.diff(df_1d['low'], prepend=df_1d['low'].iloc[0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum(
        np.maximum(df_1d['high'] - df_1d['low'], 
                   np.abs(df_1d['high'] - df_1d['close'].shift(1))),
        np.abs(df_1d['low'] - df_1d['close'].shift(1))
    )
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d = adx
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian Channel (20)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for ADX and Donchian calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and strong trend
            if price > highest_high[i] and vol > 1.5 * avg_vol[i] and strong_trend:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and strong trend
            elif price < lowest_low[i] and vol > 1.5 * avg_vol[i] and strong_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low (trend reversal)
            if price < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high (trend reversal)
            if price > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_1dADX_Volume"
timeframe = "4h"
leverage = 1.0