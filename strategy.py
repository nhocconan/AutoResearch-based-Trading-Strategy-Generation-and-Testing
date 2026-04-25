#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_AtrRegime
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter, volume confirmation, and choppiness regime filter to avoid whipsaws in ranging markets. 
- Trend filter: price > 1d EMA34 = bullish, price < 1d EMA34 = bearish (HTF trend alignment).
- In bullish 1d trend: buy breakouts above R1, sell breakdowns below S1.
- In bearish 1d trend: sell breakdowns below S1, buy breakouts above R1 (continuation logic).
- Volume confirmation: require volume > 2.0x 20-period average to avoid false breakouts.
- Choppiness regime: only trade when 4h choppiness index < 61.8 (trending market) to avoid ranging conditions.
- Exit on trend reversal or mean reversion to prior 4h Camarilla pivot (dynamic per 4h bar).
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: 1d trend filter captures major moves, volume filter reduces noise, chop filter avoids whipsaws in ranges, 4h pivot exit improves win rate.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for Camarilla pivot levels and choppiness (dynamic per 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels using previous 4h bar's OHLC
    prev_close_4h = np.roll(df_4h['close'].values, 1)
    prev_high_4h = np.roll(df_4h['high'].values, 1)
    prev_low_4h = np.roll(df_4h['low'].values, 1)
    prev_close_4h[0] = df_4h['close'].values[0]
    prev_high_4h[0] = df_4h['high'].values[0]
    prev_low_4h[0] = df_4h['low'].values[0]
    
    pivot_4h = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    range_4h = prev_high_4h - prev_low_4h
    
    # Camarilla levels for 4h
    r1_4h = pivot_4h + (range_4h * 1.1 / 12)
    s1_4h = pivot_4h - (range_4h * 1.1 / 12)
    
    # Align 4h Camarilla levels to 4h timeframe
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # Calculate 4h choppiness index (CHOP) to filter ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop = 100 * (np.log10(pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values) / np.log10(14)) / chop_denom
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    
    # Volume spike confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA(34), ATR(14), volume MA (20)
    start_idx = max(34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or
            np.isnan(pivot_4h_aligned[i]) or
            np.isnan(chop_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using EMA34
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Choppiness regime: only trade when CHOP < 61.8 (trending market)
        is_trending = chop_aligned[i] < 61.8
        
        if position == 0:
            # Breakout logic: trade in direction of 1d trend with volume confirmation and trending regime
            long_setup = (close[i] > r1_4h_aligned[i]) and htf_1d_bullish and volume_spike[i] and is_trending
            short_setup = (close[i] < s1_4h_aligned[i]) and htf_1d_bearish and volume_spike[i] and is_trending
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit on trend reversal, mean reversion to 4h pivot, or chop regime shift to ranging
            exit_signal = (not htf_1d_bullish) or (close[i] < pivot_4h_aligned[i]) or (not is_trending)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit on trend reversal, mean reversion to 4h pivot, or chop regime shift to ranging
            exit_signal = htf_1d_bullish or (close[i] > pivot_4h_aligned[i]) or (not is_trending)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_AtrRegime"
timeframe = "4h"
leverage = 1.0