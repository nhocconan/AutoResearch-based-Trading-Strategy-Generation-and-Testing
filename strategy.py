#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike and 1d chop regime filter
# - Long when price breaks above Donchian(20) high AND 12h volume > 2x 20-bar average AND 1d chop > 61.8 (ranging)
# - Short when price breaks below Donchian(20) low AND 12h volume > 2x 20-bar average AND 1d chop > 61.8 (ranging)
# - Exit when price returns to Donchian(20) midpoint or opposite breakout occurs
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25-35 trades/year (100-140 total over 4 years) to avoid fee drag
# - Donchian breakouts work well in ranging markets (2022-2025) with volume confirmation
# - Chop filter ensures we only trade in ranging regimes where mean reversion works
# - Volume confirmation filters false breakouts

name = "4h_12h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Donchian channels (20-period) on 4h data
    lookback = 20
    highest_high = pd.Series(prices['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(prices['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Pre-compute 12h volume confirmation: > 2x 20-period average
    volume_20_avg = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (2.0 * volume_20_avg)
    
    # Pre-compute 1d Chopiness Index (14-period) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d).shift()).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d).shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Max high - min low over 14 periods
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    range_14 = max_high - min_low
    
    # Chopiness Index = 100 * log10(tr_sum / range_14) / log10(14)
    chop = np.where(
        (range_14 > 0) & (tr_sum > 0),
        100 * np.log10(tr_sum / range_14) / np.log10(14),
        50  # Default when range is zero
    )
    
    # Align HTF indicators to LTF
    vol_spike_aligned = align_htf_to_ltf(prices, df_12h, vol_spike.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_20_avg[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high with volume spike and chop > 61.8 (ranging)
            if (prices['close'].iloc[i] > highest_high[i] and 
                vol_spike_aligned.iloc[i] and 
                chop_aligned[i] > 61.8):
                position = 1
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low with volume spike and chop > 61.8 (ranging)
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  vol_spike_aligned.iloc[i] and 
                  chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Price returns to Donchian midpoint
            # 2. Opposite breakout occurs
            if position == 1:
                if prices['close'].iloc[i] <= donchian_mid[i] or prices['close'].iloc[i] < lowest_low[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:
                if prices['close'].iloc[i] >= donchian_mid[i] or prices['close'].iloc[i] > highest_high[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals