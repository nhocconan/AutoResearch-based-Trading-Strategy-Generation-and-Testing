#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter and 1d volume spike confirmation
# Uses 1h price channel breakouts for entry timing, 4h EMA50 for trend alignment to avoid whipsaws
# 1d volume spike (>2.0x 20-bar average) confirms institutional participation
# ATR-based trailing stop via signal=0 when price retraces 25% of ATR from extreme
# Discrete sizing 0.20 to minimize fee drag; target 80-120 total trades over 4 years (20-30/year)
# Works in bull/bear: breakouts capture momentum, trend filter avoids counter-trend traps, volume filter ensures participation
# Session filter (08-20 UTC) reduces noise from low-liquidity periods

name = "1h_Donchian20_4hEMA50_1dVolumeSpike_v1"
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
    
    if len(df_4h) < 50 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA50 trend filter
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume spike filter (>2.0x 20-bar average)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_20)
    
    # Calculate 1h Donchian channels (20-period)
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(high_roll_20[i]) or np.isnan(low_roll_20[i]) or np.isnan(atr[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long breakout: price > 20-period high AND uptrend (price > EMA50) AND volume spike
            if close[i] > high_roll_20[i] and close[i] > ema50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
                long_extreme = close[i]
            # Short breakdown: price < 20-period low AND downtrend (price < EMA50) AND volume spike
            elif close[i] < low_roll_20[i] and close[i] < ema50_4h_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 25% of ATR from extreme
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