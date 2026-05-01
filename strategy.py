#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume spike (>1.5x 20-bar MA), and chop regime filter (CHOP(14) < 38.2 = trending)
# Long when price breaks above Donchian(20) high, price above 1d EMA50, volume spike, and CHOP < 38.2
# Short when price breaks below Donchian(20) low, price below 1d EMA50, volume spike, and CHOP < 38.2
# Uses discrete sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years (19-50/year).
# Combines price channel breakout (proven edge) with volume confirmation and regime filter to avoid whipsaws.

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ChopFilter_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    # Choppiness Index (CHOP) regime filter: CHOP(14) < 38.2 = trending (favor breakouts)
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * 14)) / log10(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    hh_ll_14 = pd.Series(high - low).rolling(window=14, min_periods=14).max().values
    chop = 100 * np.log10(sum_atr_14 / (hh_ll_14 * 14)) / np.log10(14)
    chop_trending = chop < 38.2  # Trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20, 14)  # Need 20 for Donchian, volume MA, and CHOP
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Confirmations
        vol_spike = volume_spike[i]
        is_trending = chop_trending[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian(20) high, above 1d EMA50, volume spike, trending regime
            if curr_close > highest_high[i] and curr_close > ema_1d_50_aligned[i] and vol_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian(20) low, below 1d EMA50, volume spike, trending regime
            elif curr_close < lowest_low[i] and curr_close < ema_1d_50_aligned[i] and vol_spike and is_trending:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price below Donchian(20) low or below 1d EMA50
            if curr_close < lowest_low[i] or curr_close < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price above Donchian(20) high or above 1d EMA50
            if curr_close > highest_high[i] or curr_close > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals