#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_Regime
Hypothesis: On daily timeframe, use Camarilla R1/S1 levels from previous day for breakout entries, filtered by 1w trend (EMA50), volume spike (>1.5x 20-day average), and choppy market regime (Choppiness Index > 61.8). This strategy targets ranging markets where false breakouts are faded, and trending markets where genuine breakouts are captured. Designed for 15-25 trades/year on 1d by requiring weekly alignment, volume confirmation, and regime filter. Works in both bull and bear markets by adapting to regime conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and 1w for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 5 or len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: R1 = close + 1.1*(high-low)/4, S1 = close - 1.1*(high-low)/4
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_range = prev_1d_high - prev_1d_low
    r1 = prev_1d_close + 1.1 * camarilla_range / 4
    s1 = prev_1d_close - 1.1 * camarilla_range / 4
    
    # Align Camarilla levels to 1d timeframe (no additional delay needed as they're based on completed 1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    # Choppiness Index regime filter (higher = more choppy)
    # CHOP = 100 * log10(sum(ATR over n) / (log10(n) * (max(high)-min(low) over n)))
    # Simplified: CHOP > 61.8 = ranging market (good for mean reversion at extremes)
    # We'll use a rolling version: high-low ratio vs ATR sum
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = high[0] - low[0]
    atr1 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(np.maximum(atr1 * 14, 1e-10) / np.maximum(max_high - min_low, 1e-10)) / np.log10(14)
    chop_regime = chop > 61.8  # ranging market
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need 1w EMA warmup, volume MA warmup, ATR warmup
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 1w trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # In ranging markets (chop > 61.8): fade breakouts at R1/S1
            # In trending markets (chop <= 61.8): follow breakouts with 1w trend
            if chop_regime[i]:
                # Ranging: sell at R1, buy at S1 (mean reversion)
                long_signal = (close[i] < s1_aligned[i]) and volume_spike[i]
                short_signal = (close[i] > r1_aligned[i]) and volume_spike[i]
            else:
                # Trending: follow breakouts with 1w trend
                long_signal = (close[i] > r1_aligned[i]) and trend_1w_uptrend and volume_spike[i]
                short_signal = (close[i] < s1_aligned[i]) and trend_1w_downtrend and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if chop_regime[i]:
                # In ranging market: exit at midpoint or opposite level
                midpoint = (r1_aligned[i] + s1_aligned[i]) / 2
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending market: exit if trend fails or price reverses to S1
                if not trend_1w_uptrend or close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if chop_regime[i]:
                # In ranging market: exit at midpoint or opposite level
                midpoint = (r1_aligned[i] + s1_aligned[i]) / 2
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
            else:
                # In trending market: exit if trend fails or price reverses to R1
                if not trend_1w_downtrend or close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wTrend_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0