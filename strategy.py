#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_WeeklyTrend_RegimeFilter_v1
Hypothesis: Use daily Camarilla R3/S3 breakouts with weekly trend filter and daily choppiness regime filter. 
Enters long when price breaks above R3 with volume spike in weekly uptrend and low chop regime (trending market).
Enters short when price breaks below S3 with volume spike in weekly downtrend and low chop regime.
Exits on opposite Camarilla level touch or regime shift to high chop (ranging market).
Designed for 1d timeframe to target 7-25 trades/year with discrete sizing (0.25) to minimize fee drag.
Works in bull markets via trend-following breakouts and in bear markets via regime-adaptive filtering.
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for Camarilla pivots (using previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    
    # Handle first bar (no previous day)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    # Camarilla R3 and S3 levels (wider breakout levels for fewer trades)
    # Camarilla R3 = close + 1.1*(high - low)/2
    # Camarilla S3 = close - 1.1*(high - low)/2
    camarilla_r3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_s3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Get weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        # Fallback to daily if insufficient weekly data
        close_1w_series = pd.Series(df_1d['close'].values)
    else:
        close_1w_series = pd.Series(df_1w['close'].values)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get daily data for choppiness regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # High chop (>61.8) = ranging market, Low chop (<38.2) = trending market
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    # Handle first bar for TR
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness index
    chop_denominator = highest_high_14 - lowest_low_14
    chop_denominator = np.maximum(chop_denominator, 1e-10)  # Avoid division by zero
    chop_numerator = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(chop_numerator / chop_denominator) / np.log10(14)
    
    # Regime filter: chop < 38.2 = trending (favor breakouts), chop > 61.8 = ranging (favor mean reversion)
    # We only trade breakouts in trending regime (chop < 38.2)
    trending_regime = chop < 38.2
    
    # Align all HTF data to daily timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    trending_regime_aligned = align_htf_to_ltf(prices, df_1d, trending_regime.astype(float))
    
    # Volume confirmation: volume > 2.0x 20-day average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Camarilla (1d), weekly EMA50, chop, and volume MA
    start_idx = max(2, 50, 14, 20)  # 50 for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(trending_regime_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend alignment
        trend_1w_uptrend = close[i] > ema_50_1w_aligned[i]
        trend_1w_downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Current regime is trending (low chop)
        is_trending = trending_regime_aligned[i] > 0.5
        
        if position == 0:
            # Long: price breaks above R3 + volume spike + weekly uptrend + trending regime
            long_breakout = (close[i] > camarilla_r3_aligned[i]) and \
                           (close[i-1] <= camarilla_r3_aligned[i-1]) and \
                           volume_spike[i] and \
                           trend_1w_uptrend and \
                           is_trending
            
            # Short: price breaks below S3 + volume spike + weekly downtrend + trending regime
            short_breakout = (close[i] < camarilla_s3_aligned[i]) and \
                            (close[i-1] >= camarilla_s3_aligned[i-1]) and \
                            volume_spike[i] and \
                            trend_1w_downtrend and \
                            is_trending
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches S3 OR weekly trend turns down OR regime shifts to ranging
            if (close[i] < camarilla_s3_aligned[i] or 
                not trend_1w_uptrend or 
                not is_trending):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches R3 OR weekly trend turns up OR regime shifts to ranging
            if (close[i] > camarilla_r3_aligned[i] or 
                not trend_1w_downtrend or 
                not is_trending):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_WeeklyTrend_RegimeFilter_v1"
timeframe = "1d"
leverage = 1.0