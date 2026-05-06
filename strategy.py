#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(12) zero-cross + 1d EMA50 trend + volume spike confirmation
# TRIX(12) captures momentum with smoothing to reduce noise
# 1d EMA50 provides robust trend filter (works in bull/bear: long only when price > EMA50, short when < EMA50)
# Volume spike (>2.0x 20-bar average) confirms breakout strength and participation
# ATR-based trailing stop via signal=0 when price retraces 25% of ATR from extreme
# Discrete sizing 0.25 to minimize fee drag; target 80-150 total trades over 4 years (20-38/year)
# Works in both bull/bear: TRIX zero-cross catches momentum shifts, trend filter avoids counter-trend traps, volume filter ensures follow-through

name = "4h_TRIX12_1dEMA50_VolumeSpike_v1"
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
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX(12) on 4h close: TRIX = EMA(EMA(EMA(close,12),12),12) then % change
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = 100 * (ema3.pct_change())
    trix_values = trix.values
    
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
    
    # Align HTF indicators to 4h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix_values)  # TRIX is 4h indicator, but we use 1d for HTF alignment? Wait, TRIX is calculated on 4h data, so no need to align via HTF. Actually, TRIX is LTF indicator.
    # Correction: TRIX is calculated on 4h prices, so it's already aligned. We only need to align HTF indicators (1d EMA50).
    trix_aligned = trix_values  # Already on 4h timeframe
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
        if (np.isnan(trix_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long entry: TRIX crosses above zero AND uptrend (price > EMA50) AND volume spike
            if trix_aligned[i] > 0 and trix_aligned[i-1] <= 0 and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short entry: TRIX crosses below zero AND downtrend (price < EMA50) AND volume spike
            elif trix_aligned[i] < 0 and trix_aligned[i-1] >= 0 and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
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
                signals[i] = 0.25
        elif position == -1:
            # Update short extreme
            short_extreme = min(short_extreme, close[i])
            # Exit short: price retraces 25% of ATR from extreme
            if close[i] >= short_extreme + 0.25 * atr[i]:
                signals[i] = 0.0
                position = 0
                short_extreme = 0.0
            else:
                signals[i] = -0.25
    
    return signals