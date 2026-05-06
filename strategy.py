#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses 1d Donchian channels for structure, 1d EMA50 for trend alignment, volume spike for confirmation
# Works in both bull/bear: breakouts capture momentum, trend filter avoids counter-trend traps
# Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag

name = "4h_Donchian20_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 4h timeframe (primary)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    high_touch = False
    low_touch = False
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                high_touch = False
                low_touch = False
            continue
        
        if position == 0:
            # Long breakout: price > 1d Donchian high AND uptrend (price > EMA50) AND volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                high_touch = True
            # Short breakdown: price < 1d Donchian low AND downtrend (price < EMA50) AND volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                low_touch = True
        elif position == 1:
            # Exit long: price < 1d Donchian low (reversal signal)
            if close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
                high_touch = False
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price > 1d Donchian high (reversal signal)
            if close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
                low_touch = False
            else:
                signals[i] = -0.25
    
    return signals