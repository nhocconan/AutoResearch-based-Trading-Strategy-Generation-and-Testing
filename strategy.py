#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily volume confirmation and 1w ADX trend filter.
# Buys when price breaks above 12h Donchian upper channel (20) with volume > 1.5x 20-day average and 1w ADX > 25.
# Sells when price breaks below 12h Donchian lower channel (20) with volume > 1.5x 20-day average and 1w ADX > 25.
# Uses 1d volume for confirmation and 1w ADX for trend strength to avoid whipsaws in ranging markets.
# Designed for 12h timeframe with tight entry conditions to limit trades to 50-150 over 4 years.
# Works in both bull and bear markets by requiring strong trend (ADX>25) and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Donchian channels (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 20-period Donchian channels on 12h data
    highest_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_upper = align_htf_to_ltf(prices, df_12h, highest_high)
    donchian_lower = align_htf_to_ltf(prices, df_12h, lowest_low)
    
    # Load 1d data for volume confirmation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Load 1w data for ADX trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 14-period ADX on 1w data
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_1w = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w)
    minus_di_1w = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_1w)
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = pd.Series(dx_1w).rolling(window=14, min_periods=14).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Current price and volume
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(adx_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        adx_val = adx_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_val > 25
        
        if position == 0:
            # Long conditions: break above upper channel + volume spike + strong trend
            if price > upper and vol_spike and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower channel + volume spike + strong trend
            elif price < lower and vol_spike and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: return to opposite channel or trend weakness
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to lower channel or trend weakens
                if price < lower or adx_val < 20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to upper channel or trend weakens
                if price > upper or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_1wADX_Trend"
timeframe = "12h"
leverage = 1.0