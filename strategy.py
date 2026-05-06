#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Vortex indicator with 12h EMA50 trend filter and volume spike
# Uses 12h EMA50 for trend alignment, reducing whipsaw in choppy markets
# VI+ > VI- indicates bullish momentum, VI- > VI+ indicates bearish momentum
# Volume spike (>2.0x 20-bar average) confirms breakout strength
# ATR-based trailing stop via signal=0 when price retraces 30% of ATR from extreme
# Discrete sizing 0.25 to balance profit potential and fee drag; target 80-160 total trades over 4 years (20-40/year)
# Works in both bull/bear: trend filter ensures directional alignment, volume filter ensures participation

name = "4h_Vortex_12hEMA50_VolumeSpike_v1"
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
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 trend filter
    close_12h_series = pd.Series(close_12h)
    ema50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Calculate Vortex Indicator (14)
    tr = np.maximum(np.abs(high[1:] - low[1:]), 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    vm_plus = np.abs(high[1:] - low[:-1])
    vm_minus = np.abs(low[1:] - high[:-1])
    
    # Pad arrays to match original length
    tr = np.concatenate([[np.nan], tr])
    vm_plus = np.concatenate([[np.nan], vm_plus])
    vm_minus = np.concatenate([[np.nan], vm_minus])
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    vm_plus14 = pd.Series(vm_plus).rolling(window=14, min_periods=14).sum().values
    vm_minus14 = pd.Series(vm_minus).rolling(window=14, min_periods=14).sum().values
    
    vi_plus = vm_plus14 / tr14
    vi_minus = vm_minus14 / tr14
    
    # Align HTF indicators to 4h timeframe (primary)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vi_plus_aligned = align_htf_to_ltf(prices, df_12h, vi_plus)
    vi_minus_aligned = align_htf_to_ltf(prices, df_12h, vi_minus)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vi_plus_aligned[i]) or 
            np.isnan(vi_minus_aligned[i]) or np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long: VI+ > VI- (bullish momentum) AND uptrend (price > EMA50) AND volume spike
            if vi_plus_aligned[i] > vi_minus_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short: VI- > VI+ (bearish momentum) AND downtrend (price < EMA50) AND volume spike
            elif vi_minus_aligned[i] > vi_plus_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
        elif position == 1:
            # Update long extreme
            long_extreme = max(long_extreme, close[i])
            # Exit long: price retraces 30% of ATR from extreme
            if close[i] <= long_extreme - 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 30% of ATR from extreme
            if close[i] >= short_extreme + 0.30 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals