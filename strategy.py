#!/usr/bin/env python3
# 6h_market_regime_pivot_volume_v1
# Hypothesis: 6h strategy combining market regime detection (via ADX) with daily pivot levels and volume confirmation.
# In trending regimes (ADX > 25): trade breakouts of R1/S1 in trend direction.
# In ranging regimes (ADX <= 25): fade extreme levels (R3/S3) with volume exhaustion signals.
# Uses 1d EMA50 for additional trend filter to avoid weak trends.
# Volume confirmation ensures participation. Target: 12-37 trades/year (50-150 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_market_regime_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 for trend filter (6h)
    close_s = pd.Series(close)
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ADX for regime detection (14-period)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    # Smoothed values
    tr_s = pd.Series(tr)
    atr = tr_s.ewm(span=14, min_periods=14, adjust=False).mean().values
    dm_plus_s = pd.Series(dm_plus)
    dm_minus_s = pd.Series(dm_minus)
    di_plus = 100 * dm_plus_s.ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    di_minus = 100 * dm_minus_s.ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx_s = pd.Series(dx)
    adx = dx_s.ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    # Using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pp = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (prev_high - prev_low)
    s2 = pp - (prev_high - prev_low)
    r3 = prev_high + 2 * (pp - prev_low)
    s3 = prev_low - 2 * (prev_high - pp)
    
    # Align pivot levels to 6h timeframe (completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema50[i]) or np.isnan(adx[i]) or np.isnan(volume_ma[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * volume_ma[i]
        
        # Regime detection
        is_trending = adx[i] > 25
        
        if position == 1:  # Long position
            # Exit conditions
            exit_signal = False
            if is_trending:
                # In trend: exit if price crosses below S1 or ADX weakens
                if close[i] < s1_aligned[i] or adx[i] < 20:
                    exit_signal = True
            else:
                # In range: exit if price reaches R3 or shows weakness
                if close[i] >= r3_aligned[i] or (close[i] < close[i-1] and volume[i] < volume[i-1]):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_signal = False
            if is_trending:
                # In trend: exit if price crosses above R1 or ADX weakens
                if close[i] > r1_aligned[i] or adx[i] < 20:
                    exit_signal = True
            else:
                # In range: exit if price reaches S3 or shows weakness
                if close[i] <= s3_aligned[i] or (close[i] > close[i-1] and volume[i] < volume[i-1]):
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if is_trending:
                # Trending regime: breakout of R1/S1 with volume and EMA50 filter
                if (close[i] > r1_aligned[i] and close[i] > ema50[i] and volume_confirmed):
                    position = 1
                    signals[i] = 0.25
                elif (close[i] < s1_aligned[i] and close[i] < ema50[i] and volume_confirmed):
                    position = -1
                    signals[i] = -0.25
            else:
                # Ranging regime: fade extreme levels (R3/S3) with volume exhaustion
                # Long near S3 with selling exhaustion (price down but volume down)
                if (close[i] <= s3_aligned[i] * 1.005 and  # Allow small buffer
                    close[i] < close[i-1] and 
                    volume[i] < volume[i-1] and
                    volume_confirmed):
                    position = 1
                    signals[i] = 0.25
                # Short near R3 with buying exhaustion (price up but volume down)
                elif (close[i] >= r3_aligned[i] * 0.995 and  # Allow small buffer
                      close[i] > close[i-1] and
                      volume[i] < volume[i-1] and
                      volume_confirmed):
                    position = -1
                    signals[i] = -0.25
    
    return signals