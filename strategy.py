#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
    # Designed for low trade frequency (19-50/year) to minimize fee drag
    # Works in trending markets via breakouts; chop filter avoids whipsaws in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF volume and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ATR for chop regime (ATR(14))
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d true range average for chop (using TR)
    tr_avg_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d high-low range for chop
    hl_range_1d = high_1d - low_1d
    hl_avg_14_1d = pd.Series(hl_range_1d).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14) / (sum(HL14) * 14)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    sum_hl_14 = pd.Series(hl_range_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (sum_hl_14 * 14)) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Align HTF indicators to 4h
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(vol_avg_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period 1d volume average
        # Use 4h volume but compare to 1d average volume (scaled)
        volume_confirmed = volume[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Regime filter: only trade when trending (CHOP < 38.2)
        trending_regime = chop_aligned[i] < 38.2
        
        # Donchian breakout conditions
        breakout_long = close[i] > high_20[i-1]  # Break above 20-period high
        breakout_short = close[i] < low_20[i-1]  # Break below 20-period low
        
        # Entry conditions: breakout + volume + trending regime
        enter_long = breakout_long and volume_confirmed and trending_regime
        enter_short = breakout_short and volume_confirmed and trending_regime
        
        # Exit conditions: opposite Donchian breakout or middle line
        donchian_mid = (high_20[i] + low_20[i]) / 2
        exit_long = position == 1 and (close[i] < donchian_mid or breakout_short)
        exit_short = position == -1 and (close[i] > donchian_mid or breakout_long)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0