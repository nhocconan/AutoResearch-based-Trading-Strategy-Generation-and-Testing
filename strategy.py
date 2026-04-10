#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and chop regime filter
# - Donchian(20) breakout on 12h: long when price > highest high of last 20 bars, short when price < lowest low
# - Volume confirmation: current 12h volume > 1.3x 20-period average volume
# - Regime filter: Choppiness Index(14) on 1d between 38.2 and 61.8 (avoid extreme trending/choppy)
# - ATR(14) trailing stop (2.5x) on 12h timeframe
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year (50-100 total over 4 years) to stay within HARD MAX: 200 total
# - Donchian breakouts capture strong momentum moves, volume confirms legitimacy, chop filter avoids false signals in extreme regimes

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum.reduce([tr1, tr2, tr3])
    
    # Sum of True Range over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(tr_sum_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero/negative
    hl_range = hh_14 - ll_14
    chop_raw = np.where((tr_sum_14 > 0) & (hl_range > 0), 
                        tr_sum_14 / hl_range, np.nan)
    chop = 100 * np.log10(chop_raw) / np.log10(14)
    
    # Pre-compute 1d volume and its 20-period moving average for volume confirmation
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Pre-compute 12h Donchian channels (20-period)
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Donchian upper band: highest high of last 20 periods
    donchian_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band: lowest low of last 20 periods
    donchian_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 12h ATR for trailing stop
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr1_12h[0] = np.nan
    tr2_12h[0] = np.nan
    tr3_12h[0] = np.nan
    tr_12h = np.maximum.reduce([tr1_12h, tr2_12h, tr3_12h])
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 12h volume and its 20-period moving average
    volume_12h = prices['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop
    lowest_since_entry = 0.0   # for trailing stop
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(chop_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(atr_12h[i]) or np.isnan(volume_ma_20_12h[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get current 12h volume for filter
        volume_12h_current = volume_12h[i]
        
        # Get current 12h close
        close_price = close_12h[i]
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_spike = volume_12h_current > 1.3 * volume_ma_20_12h[i]
        
        # Chop regime filter: avoid extreme values (<38.2 = too trending, >61.8 = too choppy)
        chop_value = chop_aligned[i]
        chop_filter = (chop_value >= 38.2) & (chop_value <= 61.8)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band + volume spike + chop filter
            if close_price > donchian_upper[i] and volume_spike and chop_filter:
                position = 1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                highest_since_entry = prices['high'].iloc[i]
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band + volume spike + chop filter
            elif close_price < donchian_lower[i] and volume_spike and chop_filter:
                position = -1
                entry_price = prices['open'].iloc[i+1] if i+1 < n else prices['close'].iloc[i]
                lowest_since_entry = prices['low'].iloc[i]
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit or trailing stop
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, prices['high'].iloc[i])
                # ATR trailing stop: exit when price drops 2.5*ATR from highest point
                trailing_stop = prices['close'].iloc[i] < highest_since_entry - 2.5 * atr_12h[i]
                exit_condition = trailing_stop
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, prices['low'].iloc[i])
                # ATR trailing stop: exit when price rises 2.5*ATR from lowest point
                trailing_stop = prices['close'].iloc[i] > lowest_since_entry + 2.5 * atr_12h[i]
                exit_condition = trailing_stop
            
            if exit_condition:
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals