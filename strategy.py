#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; in ranging markets (CHOP > 50) it tends to revert
# 1d EMA50 provides trend filter to avoid counter-trend trades during strong trends
# Volume confirmation (>1.8x 20-bar average) ensures breakout/mean reversion has participation
# ATR-based trailing stop via signal=0 when price retraces 25% of ATR from extreme
# Discrete sizing 0.25 to balance profit potential and fee drag; target 80-150 total trades over 4 years (20-38/year)
# Works in both bull/bear: mean reversion captures reversals in ranges, trend filter avoids whipsaw in trends

name = "4h_WilliamsR_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams %R(14) on 4h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>1.8x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Calculate Choppiness Index(14) for regime filter
    # CHOP > 50 indicates ranging market (good for mean reversion)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_min_range = pd.Series(high).rolling(window=14, min_periods=14).max().values - \
                    pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / np.log10(14) / max_min_range)
    # Handle division by zero and invalid values
    chop = np.where((max_min_range == 0) | np.isnan(atr_sum) | np.isnan(max_min_range), 50, chop)
    
    # Align HTF indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or np.isnan(chop[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) AND ranging market (CHOP > 50) AND volume spike
            # Avoid counter-trend: only long if price > EMA50 (mild uptrend bias) OR strong ranging
            if williams_r[i] < -80 and chop[i] > 50 and volume_filter[i]:
                # Additional trend filter: allow long in mild uptrend or strong range
                if close[i] > ema50_1d_aligned[i] or chop[i] > 60:  # Either uptrend or strong ranging
                    signals[i] = 0.25
                    position = 1
                    long_extreme = close[i]
            # Short entry: Williams %R overbought (> -20) AND ranging market (CHOP > 50) AND volume spike
            # Avoid counter-trend: only short if price < EMA50 (mild downtrend bias) OR strong ranging
            elif williams_r[i] > -20 and chop[i] > 50 and volume_filter[i]:
                # Additional trend filter: allow short in mild downtrend or strong range
                if close[i] < ema50_1d_aligned[i] or chop[i] > 60:  # Either downtrend or strong ranging
                    signals[i] = -0.25
                    position = -1
                    short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 25% of ATR from extreme OR Williams %R becomes overbought
            if close[i] <= long_extreme - 0.25 * atr[i] or williams_r[i] > -20:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 25% of ATR from extreme OR Williams %R becomes oversold
            if close[i] >= short_extreme + 0.25 * atr[i] or williams_r[i] < -80:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals