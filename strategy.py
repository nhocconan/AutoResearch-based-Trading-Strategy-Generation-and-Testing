#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data for regime filter (choppiness)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 14-period weekly ATR for choppiness
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range calculation
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 14-period weekly high-low range
    hh_14w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    sum_atr_14w = pd.Series(atr_14w).rolling(window=14, min_periods=14).sum().values
    range_14w = hh_14w - ll_14w
    chop = np.where(range_14w > 0, 100 * np.log10(sum_atr_14w / range_14w) / np.log10(14), 50)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Calculate 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Price array
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period average volume for volume filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        ema34_1d_val = ema34_1d_aligned[i]
        chop_val = chop_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price above/below 1d EMA34
        uptrend = price > ema34_1d_val
        downtrend = price < ema34_1d_val
        
        # Chop filter: chop > 50 indicates ranging market (mean reversion opportunity)
        chop_filter = chop_val > 50
        
        # Volume filter: current volume > 1.3 * 20-period average volume
        vol_spike = vol > 1.3 * vol_ma
        
        if position == 0:
            # Long: price breaks above 12h Donchian high + 1d uptrend + chop filter + volume spike
            if price > donch_high_val and uptrend and chop_filter and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low + 1d downtrend + chop filter + volume spike
            elif price < donch_low_val and downtrend and chop_filter and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through opposite Donchian level or chop drops or volume drops
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below Donchian low or chop < 40 (trending) or volume drop
                if price < donch_low_val or chop_val < 40 or not vol_spike:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above Donchian high or chop < 40 (trending) or volume drop
                if price > donch_high_val or chop_val < 40 or not vol_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_ChopFilter_VolSpike"
timeframe = "12h"
leverage = 1.0