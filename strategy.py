#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volume spike
# Donchian breakouts capture strong momentum moves. EMA50 on daily ensures alignment with
# primary trend to avoid counter-trend trades. Volume spike confirmation filters false breakouts.
# ATR-based position sizing adjusts for volatility. Designed for 20-40 trades/year to minimize fee drag.
# Works in both bull and bear markets by following the primary trend on higher timeframe.

name = "4h_Donchian20_Breakout_1dEMA50_Trend_ATR_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14) for volatility-based position sizing and stop consideration
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, lookback, 14, 20)  # EMA50, Donchian20, ATR14, VolumeMA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # ATR-based position size (volatility adjusted)
        # Base size 0.25, scaled by ATR relative to price
        atr_ratio = atr[i] / close[i] if close[i] > 0 else 0
        # Normalize ATR ratio to 0.01-0.05 range (typical for crypto)
        size_multiplier = np.clip(atr_ratio * 50, 0.5, 1.5)  # Scale to 0.5-1.5
        base_size = 0.25
        position_size = base_size * size_multiplier
        # Cap at 0.35 max
        position_size = min(position_size, 0.35)
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian high, volume spike, uptrend
            if close[i] > highest_high[i] and vol_spike and uptrend:
                signals[i] = position_size
                position = 1
            # Short: break below Donchian low, volume spike, downtrend
            elif close[i] < lowest_low[i] and vol_spike and downtrend:
                signals[i] = -position_size
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below Donchian low or trend reversal
            if close[i] < lowest_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        
        elif position == -1:  # Short position
            # Exit on break above Donchian high or trend reversal
            if close[i] > highest_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals