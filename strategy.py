#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeConfirm_ChopRegime_v1
Hypothesis: Donchian(20) breakout on 4h aligned with 1d trend (close vs EMA50) and volume spike (>1.5x average) during non-choppy markets (Choppiness Index < 38.2) captures strong directional moves while avoiding false breakouts in ranging markets. Works in bull/bear via 1d trend filter. Discrete sizing 0.25 to control risk and minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for volatility and choppiness calculation
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of True Range over 14 periods for Choppiness Index
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods for Choppiness Index
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CI = 100 * log10(tr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # small epsilon to prevent div by zero
    log_tr_sum = np.log10(np.where(tr_sum > 0, tr_sum, 1e-10))
    log_range = np.log10(range_hl)
    chop = 100 * (log_tr_sum / log_range) / np.log10(14)
    
    # Average volume for confirmation (24-period SMA = 6h * 4 = 24 periods on 4h = 1d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of Donchian(20), EMA(50), ATR(14), volume(24), chop(14)
    start_idx = max(20, 50, 14, 24, 14)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        chop_val = chop[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(atr_val) or 
            np.isnan(chop_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Donchian(20) breakout levels (using previous 20 periods)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclusive, so we use up to i-1
        if lookback_end - lookback_start < 20:
            # Not enough lookback, hold
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Trend filter: price vs 1d EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Chop regime: only trade when market is trending (CHOP < 38.2)
        trending_regime = chop_val < 38.2
        
        # Long: price CLOSES above Donchian high with 1d uptrend, volume, and trending regime
        long_condition = (close_val > highest_high) and uptrend and volume_confirmed and trending_regime
        # Short: price CLOSES below Donchian low with 1d downtrend, volume, and trending regime
        short_condition = (close_val < lowest_low) and downtrend and volume_confirmed and trending_regime
        
        # Exit: price retests opposite Donchian level or volatility-based stop
        long_exit = (position == 1 and close_val <= lowest_low)
        short_exit = (position == -1 and close_val >= highest_high)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeConfirm_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0