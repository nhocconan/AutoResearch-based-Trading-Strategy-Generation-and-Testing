#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v4
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 1d ATR-based trend filter and volume spike confirmation.
- Trend filter: price > 1d close + 0.5*ATR(14) = bullish, price < 1d close - 0.5*ATR(14) = bearish, else ranging.
- In trending markets: buy breakouts above R1, sell breakdowns below S1.
- In ranging markets: fade extremes at R1/S1 with mean reversion to pivot.
- Volume confirmation: require volume > 1.5x 20-period average to avoid false breakouts.
- Position size: 0.25. Target: 75-200 total trades over 4 years = 19-50/year.
- Works in both bull and bear: ATR trend filter adapts to volatility regime, volume filters noise.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(14) using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily Camarilla pivot levels (using previous day's OHLC)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
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
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Volume spike confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14) and volume MA (20)
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(pivot_aligned[i]) or
            np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend using ATR bands
        htf_1d_bullish = close[i] > close_1d_aligned[i] + (0.5 * atr_14_1d_aligned[i])
        htf_1d_bearish = close[i] < close_1d_aligned[i] - (0.5 * atr_14_1d_aligned[i])
        
        # Determine if we are in trending or ranging market based on ATR bands
        trending_market = htf_1d_bullish or htf_1d_bearish
        ranging_market = not trending_market
        
        if position == 0:
            if trending_market:
                # Trending market: trade breakout continuation
                long_setup = (close[i] > r1_aligned[i]) and htf_1d_bullish and volume_spike[i]
                short_setup = (close[i] < s1_aligned[i]) and htf_1d_bearish and volume_spike[i]
            else:
                # Ranging market: trade mean reversion at extremes
                long_setup = (close[i] < s1_aligned[i]) and (close[i] > s3_aligned[i]) and volume_spike[i]  # Oversold bounce
                short_setup = (close[i] > r1_aligned[i]) and (close[i] < r3_aligned[i]) and volume_spike[i]  # Overbought rejection
            
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
            else:
                # In ranging market: exit on mean reversion to pivot or touch of R1
                exit_signal = (close[i] > pivot_aligned[i]) or (close[i] > r1_aligned[i])
            
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
            else:
                # In ranging market: exit on mean reversion to pivot or touch of S1
                exit_signal = (close[i] < pivot_aligned[i]) or (close[i] < s1_aligned[i])
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dATR_Trend_VolumeSpike_v4"
timeframe = "4h"
leverage = 1.0