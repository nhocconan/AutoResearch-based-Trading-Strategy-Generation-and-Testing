#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversal with 12h volume spike and chop regime filter
# Long when Williams %R(14) < -80 (oversold) AND 12h volume > 2.0x 20-period volume SMA AND chop > 61.8 (ranging market)
# Short when Williams %R(14) > -20 (overbought) AND 12h volume > 2.0x 20-period volume SMA AND chop > 61.8 (ranging market)
# Uses 12h Williams %R for extreme reversal signals, volume spike for conviction, chop filter to avoid trending markets
# Works in ranging markets (chop > 61.8) where mean reversion is effective, avoids strong trends
# Discrete sizing 0.25 limits drawdown; targets 30-60 trades/year to balance opportunity and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop for Williams %R, volume, and chop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicator: Williams %R (14-period) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 12h Indicator: Choppiness Index (14-period) for regime filter ===
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh_14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum(TR)/ (HH - LL)) / log10(14)
    # Avoid division by zero and log of zero/negative
    hh_minus_ll = hh_14 - ll_14
    chop_raw = np.where((hh_minus_ll > 0) & (tr_sum_14 > 0), 
                        100 * np.log10(tr_sum_14 / hh_minus_ll) / np.log10(14), 
                        50)  # neutral value when undefined
    
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_raw)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100  # Need 50 for chop calculation, 20 for volume SMA, 14 for Williams %R
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(vol_sma_20_12h_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 2.0x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 2.0
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Chop filter: > 61.8 indicates ranging market (good for mean reversion)
        chop_filter = chop_aligned[i] > 61.8
        
        # Price levels
        price = close[i]
        
        # === LONG CONDITIONS ===
        # Williams %R < -80 (oversold) AND volume confirmation AND chop regime
        if (williams_r_aligned[i] < -80) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Williams %R > -20 (overbought) AND volume confirmation AND chop regime
        elif (williams_r_aligned[i] > -20) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_WilliamsR14_12hVolume2x_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0