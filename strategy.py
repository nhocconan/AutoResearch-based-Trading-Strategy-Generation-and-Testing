#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_ChopFilter_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d EMA34 trend filter and choppiness regime filter.
- In trending markets (CHOP < 38.2): buy breakouts above R1, sell breakdowns below S1.
- In ranging markets (CHOP > 61.8): fade extremes at R1/S1 with mean reversion to pivot.
- Volume confirmation: require volume > 1.3x 20-period average to avoid false breakouts.
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: regime filter adapts to market conditions, volume filters noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(df_1d['high'].values, 1)
    prev_low = np.roll(df_1d['low'].values, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = df_1d['high'].values[0]
    prev_low[0] = df_1d['low'].values[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume spike confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * vol_ma_20)
    
    # Calculate Choppiness Index (CHOP) on 1d timeframe for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (n * (max(high) - min(low)))) / log10(n)
    # Simplified: CHOP = 100 * log10(ATR_sum / (n * range)) / log10(n)
    # We'll use a rolling window of 14 days
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.abs(high_1d[0] - low_1d[0])], tr])  # first TR
    
    # ATR(14) - smoothed TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP for 14-period
    chop_raw = np.full(len(df_1d), np.nan)
    for i in range(14, len(df_1d)):
        atr_sum = np.sum(atr_14[i-13:i+1])  # sum of last 14 ATR values
        period_high = np.max(high_1d[i-13:i+1])
        period_low = np.min(low_1d[i-13:i+1])
        period_range = period_high - period_low
        if period_range > 0 and atr_sum > 0:
            chop_raw[i] = 100 * np.log10(atr_sum / (14 * period_range)) / np.log10(14)
        else:
            chop_raw[i] = 50  # neutral if calculation invalid
    
    # Align CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), and CHOP (14)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above 1d EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Determine market regime based on Choppiness Index
        # CHOP < 38.2 = trending, CHOP > 61.8 = ranging, 38.2-61.8 = transitional
        trending_market = chop_aligned[i] < 38.2
        ranging_market = chop_aligned[i] > 61.8
        
        if position == 0:
            if trending_market:
                # Trending market: trade breakout continuation
                long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_spike[i]
                short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_spike[i]
            elif ranging_market:
                # Ranging market: trade mean reversion at extremes
                long_setup = (close[i] < s1_aligned[i]) and (close[i] > s3_aligned[i]) and volume_spike[i]  # Oversold bounce
                short_setup = (close[i] > r1_aligned[i]) and (close[i] < r3_aligned[i]) and volume_spike[i]  # Overbought rejection
            else:
                # Transitional market: no trading
                long_setup = False
                short_setup = False
            
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
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of S1
                exit_signal = (not htf_1d_bullish) or (close[i] < s1_aligned[i])
            elif ranging_market:
                # In ranging market: exit on mean reversion to pivot or touch of R1
                exit_signal = (close[i] > pivot_aligned[i]) or (close[i] > r1_aligned[i])
            else:
                # Transitional market: exit on mean reversion to pivot
                exit_signal = close[i] > pivot_aligned[i]
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if trending_market:
                # In trending market: exit on trend reversal or touch of R1
                exit_signal = htf_1d_bullish or (close[i] > r1_aligned[i])
            elif ranging_market:
                # In ranging market: exit on mean reversion to pivot or touch of S1
                exit_signal = (close[i] < pivot_aligned[i]) or (close[i] < s1_aligned[i])
            else:
                # Transitional market: exit on mean reversion to pivot
                exit_signal = close[i] < pivot_aligned[i]
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0