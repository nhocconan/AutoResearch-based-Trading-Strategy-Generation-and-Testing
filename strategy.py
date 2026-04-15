#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1w RSI trend filter
# Uses weekly RSI to determine trend direction (bull/bear), then only takes long breakouts in bull markets
# and short breakouts in bear markets. This avoids whipsaw in ranging markets. Volume confirmation ensures
# breakouts are genuine. Designed to work in both bull and bear by adapting to weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data (primary timeframe) for price action
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Load 1w data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donch_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate RSI (14-period) on 1w
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h := np.zeros(len(high_12h))).rolling(window=20, min_periods=20).mean().values
    # Actually compute volume average properly
    vol_sum = np.zeros(len(high_12h))
    for i in range(len(high_12h)):
        if i < 20:
            vol_sum[i] = np.sum(high_12h[:i+1]) if i == 0 else vol_sum[i-1] + high_12h[i]  # placeholder
    # Correct volume average calculation
    vol_avg_12h = pd.Series([0]*len(high_12h)).rolling(window=20, min_periods=20).mean().values  # dummy init
    vol_series = pd.Series(volume[:len(high_12h)*12 if len(high_12h)*12 <= len(volume) else len(volume)]).rolling(window=20, min_periods=20).mean()
    # Simpler: use 12h volume from resampled data isn't available, so use 1h volume proxy via scaling
    # Instead, calculate volume average on available 12h-aligned volume
    vol_12h = np.zeros(len(high_12h))
    # Since we don't have 12h volume directly, we'll use 1h volume and resample conceptually via averaging
    # But per rules, we must use actual data. Let's use 1d volume as proxy for 12h trend
    # Actually, we can get volume from 12h bars if we had it, but we don't. So we'll skip volume filter
    # and rely on price breakout + trend filter
    
    # Revert to using 1d volume for confirmation as it's available
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    donch_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_high_12h)
    donch_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donch_low_12h)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_12h_aligned[i]) or np.isnan(donch_low_12h_aligned[i]) or
            np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian high + volume spike + weekly RSI > 50 (bullish)
        if (close[i] > donch_high_12h_aligned[i] and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            rsi_1w_aligned[i] > 50 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price breaks below Donchian low + volume spike + weekly RSI < 50 (bearish)
        elif (close[i] < donch_low_12h_aligned[i] and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              rsi_1w_aligned[i] < 50 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or weekly RSI crosses 50 (trend change)
        elif position == 1 and (close[i] < donch_low_12h_aligned[i] or rsi_1w_aligned[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > donch_high_12h_aligned[i] or rsi_1w_aligned[i] > 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Volume_WeeklyRSI"
timeframe = "12h"
leverage = 1.0