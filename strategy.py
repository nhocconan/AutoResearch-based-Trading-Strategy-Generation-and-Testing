#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + Volume Spike + Choppiness Regime Filter
# Long when TRIX crosses above zero AND volume > 2.0x 20-bar avg AND Choppiness Index < 38.2 (trending)
# Short when TRIX crosses below zero AND volume > 2.0x 20-bar avg AND Choppiness Index < 38.2
# Exit when TRIX crosses zero in opposite direction
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 19-50 trades/year on 4h timeframe.
# TRIX is a momentum oscillator that filters noise and identifies trend changes.
# Volume confirmation ensures breakout strength. Choppiness filter avoids ranging markets.
# This combination has worked well on ETH historically and should generalize to BTC.

name = "4h_TRIX_VolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (15-period EMA of EMA of EMA of ROC)
    # ROC = (close - close_prev) / close_prev * 100
    close_series = pd.Series(close)
    roc = close_series.pct_change() * 100
    ema1 = roc.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Choppiness Index (14-period) - measures whether market is choppy (ranging) or trending
    # CHOP = 100 * log10(sum(ATR) / (log(n) * (highest_high - lowest_low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = np.log10(14) * (hh - ll)
    chop_raw = np.where(denominator != 0, 
                        np.sum(pd.Series(atr).rolling(window=14, min_periods=14).sum().values) / denominator,
                        100)
    chop = 100 * np.log10(np.maximum(chop_raw, 1e-10)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)  # Replace NaN with neutral value
    
    # Trending regime: CHOP < 38.2
    trending_regime = chop < 38.2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(45, 20, 14)  # TRIX warmup + volume MA + chop warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_trix = trix[i]
        curr_trix_prev = trix[i-1] if i > 0 else 0
        is_trending = trending_regime[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if curr_trix <= 0 and curr_trix_prev > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if curr_trix >= 0 and curr_trix_prev < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when TRIX crosses above zero AND volume confirmation AND trending regime
            if curr_trix > 0 and curr_trix_prev <= 0 and vol_conf and is_trending:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero AND volume confirmation AND trending regime
            elif curr_trix < 0 and curr_trix_prev >= 0 and vol_conf and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals