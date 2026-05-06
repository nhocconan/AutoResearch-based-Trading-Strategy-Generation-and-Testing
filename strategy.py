#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 4h Donchian(20) for structure, 1d EMA50 for trend alignment (reduces whipsaw in bear markets)
# Volume spike (>2.0x 20-bar average) confirms breakout strength
# ATR-based trailing stop via signal=0 when price retraces 25% of ATR from extreme
# Discrete sizing 0.20 to minimize fee drag; target 60-150 total trades over 4 years (15-37/year)
# Works in both bull/bear: breakouts capture momentum, trend filter avoids counter-trend traps, volume filter ensures participation
# Session filter (08-20 UTC) reduces noise trades

name = "1h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "1h"
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
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian(20) - upper and lower bands
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_high_4h = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_low_4h = low_4h_series.rolling(window=20, min_periods=20).min().values
    
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
    
    # Align HTF indicators to 1h timeframe
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
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
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > 4h Donchian high AND uptrend (price > 1d EMA50) AND volume spike
            if close[i] > donchian_high_4h_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.20
                position = 1
                long_extreme = close[i]
            # Short breakdown: price < 4h Donchian low AND downtrend (price < 1d EMA50) AND volume spike
            elif close[i] < donchian_low_4h_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.20
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 25% of ATR from extreme (tighter stop for 1h)
            if close[i] <= long_extreme - 0.25 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 25% of ATR from extreme
            if close[i] >= short_extreme + 0.25 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.20
    
    return signals