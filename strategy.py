#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout + volume confirmation
# Uses Choppiness Index(14) to filter regimes: >61.8 = ranging (mean revert), <38.2 = trending (trend follow)
# In ranging markets: fade Donchian breaks (sell upper, buy lower) with volume confirmation
# In trending markets: follow Donchian breaks (buy upper, sell lower) with volume confirmation
# Volume > 1.5x 20-period average confirms breakout strength
# Target: 20-30 trades/year to minimize fee decay while capturing both trending and ranging markets
# Focus on BTC/ETH as primary assets with proven regime effectiveness from research

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_period = 14
    atr = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # Sum of ATR over 14 periods
    sum_atr = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        sum_atr[i] = np.nansum(atr[i-atr_period+1:i+1])
    
    # Highest high and lowest low over 14 periods
    max_hh = np.full(len(close_1d), np.nan)
    min_ll = np.full(len(close_1d), np.nan)
    for i in range(atr_period, len(close_1d)):
        max_hh[i] = np.nanmax(high_1d[i-atr_period+1:i+1])
        min_ll[i] = np.nanmin(low_1d[i-atr_period+1:i+1])
    
    # Choppiness Index
    chop = np.full(len(close_1d), 50.0)  # default neutral
    for i in range(atr_period, len(close_1d)):
        if max_hh[i] != min_ll[i] and sum_atr[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (max_hh[i] - min_ll[i])) / np.log10(atr_period)
    
    chop_align = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Donchian(20) on 4h data
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(donchian_period, n):
        upper[i] = np.max(high[i-donchian_period:i])
        lower[i] = np.min(low[i-donchian_period:i])
    
    # 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(donchian_period, vol_period, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(chop_align[i]) or
            np.isnan(upper[i]) or
            np.isnan(lower[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Determine regime from Choppiness Index
        ranging = chop_align[i] > 61.8
        trending = chop_align[i] < 38.2
        
        # Volume confirmation: spike > 1.5x average
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            if ranging:
                # In ranging markets: mean reversion at Donchian bands
                if price > upper[i] and volume_confirmation:
                    signals[i] = -size  # sell at upper band
                    position = -1
                elif price < lower[i] and volume_confirmation:
                    signals[i] = size   # buy at lower band
                    position = 1
                else:
                    signals[i] = 0.0
            elif trending:
                # In trending markets: follow breakouts
                if price > upper[i] and volume_confirmation:
                    signals[i] = size   # buy breakout
                    position = 1
                elif price < lower[i] and volume_confirmation:
                    signals[i] = -size  # sell breakdown
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop: no position
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price returns to middle of range or breaks lower band
            if ranging and price < (upper[i] + lower[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price returns to middle of range or breaks upper band
            if ranging and price > (upper[i] + lower[i]) / 2:
                signals[i] = 0.0
                position = 0
            elif price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Chop_Regime_Donchian_Breakout_Volume"
timeframe = "4h"
leverage = 1.0