#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with Donchian(20) breakout and volume confirmation
# Uses 1d Choppiness Index to identify trending vs ranging markets (trend when CHOP < 38.2, range when CHOP > 61.8)
# In trending regimes: enter on Donchian(20) breakouts in direction of trend
# In ranging regimes: enter on mean reversion at Donchian channels
# Volume confirmation (2x 20-period average) filters false breakouts
# Target: 20-30 trades/year to minimize fee decay while capturing both trending and ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for Choppiness Index
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr = np.zeros(len(close_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    # ATR(14) using Wilder's smoothing
    atr_period = 14
    atr_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= atr_period:
        atr_1d[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(close_1d)):
            atr_1d[i] = (atr_1d[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate Choppiness Index
    chop_period = 14
    chop_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= chop_period:
        for i in range(chop_period-1, len(close_1d)):
            # Sum of true ranges over chop_period
            tr_sum = np.sum(tr[i-chop_period+1:i+1])
            # Highest high and lowest low over chop_period
            hh = np.max(high_1d[i-chop_period+1:i+1])
            ll = np.min(low_1d[i-chop_period+1:i+1])
            if hh != ll and tr_sum > 0:
                chop_1d[i] = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(chop_period)
            else:
                chop_1d[i] = 50.0  # neutral when no range
    
    # Calculate 1d EMA34 for trend direction
    ema_len = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_len:
        multiplier = 2 / (ema_len + 1)
        ema_1d[ema_len-1] = np.mean(close_1d[:ema_len])
        for i in range(ema_len, len(close_1d)):
            ema_1d[i] = (close_1d[i] * multiplier) + (ema_1d[i-1] * (1 - multiplier))
    
    # Align 1d indicators to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels on 12h
    donch_period = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(donch_period-1, n):
        donch_high[i] = np.max(high[i-donch_period+1:i+1])
        donch_low[i] = np.min(low[i-donch_period+1:i+1])
    
    # Calculate 20-period average volume for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(50, donch_period, vol_period) + 2
    
    for i in range(start_idx, n):
        if (np.isnan(chop_1d_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume confirmation: at least 2x average volume
        volume_confirmation = vol_ratio > 2.0
        
        chop = chop_1d_aligned[i]
        ema = ema_1d_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        if position == 0:
            # Determine market regime
            if chop < 38.2:  # Trending regime
                # Long: Donchian breakout above upper band with uptrend and volume
                if price > upper and price > ema and volume_confirmation:
                    signals[i] = size
                    position = 1
                # Short: Donchian breakdown below lower band with downtrend and volume
                elif price < lower and price < ema and volume_confirmation:
                    signals[i] = -size
                    position = -1
            elif chop > 61.8:  # Ranging regime
                # Long: Mean reversion at lower band with volume
                if price <= lower and volume_confirmation:
                    signals[i] = size
                    position = 1
                # Short: Mean reversion at upper band with volume
                elif price >= upper and volume_confirmation:
                    signals[i] = -size
                    position = -1
            else:  # Transition regime - no trades
                signals[i] = 0.0
        elif position == 1:
            # Long exit conditions
            if chop < 38.2:  # Trending regime
                # Exit on trend reversal or Donchian breakdown
                if price < ema or price < lower:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:  # Ranging or transition regime
                # Exit on mean reversion to middle or opposite band
                if price >= (upper + lower) / 2 or price >= upper:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
        elif position == -1:
            # Short exit conditions
            if chop < 38.2:  # Trending regime
                # Exit on trend reversal or Donchian breakout
                if price > ema or price > upper:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:  # Ranging or transition regime
                # Exit on mean reversion to middle or opposite band
                if price <= (upper + lower) / 2 or price <= lower:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
    
    return signals

name = "12h_Choppiness_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0