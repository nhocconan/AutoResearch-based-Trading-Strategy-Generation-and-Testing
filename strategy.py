#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX momentum with 1w ADX regime filter and volume confirmation
# TRIX(12) crossing zero line captures momentum shifts with reduced whipsaw vs MACD
# 1w ADX > 25 filters for trending markets only, avoiding range-bound losses
# Volume spike (>1.8x 50-period EMA volume) confirms institutional participation
# Discrete sizing 0.25 targets 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in bull markets (long momentum with uptrend) and bear markets (short momentum with downtrend)
# 12h timeframe minimizes fee drag while capturing multi-day momentum moves

name = "12h_TRIX_1wADX_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # Need enough data for ADX calculation
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w ADX(14) for regime filter
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - prev_close_1w)
    tr3 = np.abs(low_1w - prev_close_1w)
    tr_1w = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # +DM and -DM
    high_diff = np.diff(high_1w, prepend=np.nan)
    low_diff = -np.diff(low_1w, prepend=np.nan)
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder's smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + data[i]
                else:
                    result[i] = np.nan
        return result
    
    atr_1w = wilders_smoothing(tr_1w, 14)
    plus_di_1w = 100 * wilders_smoothing(plus_dm, 14) / atr_1w
    minus_di_1w = 100 * wilders_smoothing(minus_dm, 14) / atr_1w
    dx_1w = 100 * np.abs(plus_di_1w - minus_di_1w) / (plus_di_1w + minus_di_1w)
    adx_1w = wilders_smoothing(dx_1w, 14)
    
    # Align HTF ADX to 12h timeframe (wait for completed 1w bar)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate TRIX(12,9,9) from 12h close prices
    # TRIX = EMA(EMA(EMA(close, 12), 9), 9) - 1 period ago, then / previous value * 100
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = np.nan  # First value undefined
    
    # Volume confirmation: 50-period EMA of volume
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trix[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: TRIX > 0 (bullish momentum) AND ADX > 25 (trending) AND volume spike
            if trix[i] > 0 and adx_1w_aligned[i] > 25 and volume[i] > (1.8 * vol_ema_50[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX < 0 (bearish momentum) AND ADX > 25 (trending) AND volume spike
            elif trix[i] < 0 and adx_1w_aligned[i] > 25 and volume[i] > (1.8 * vol_ema_50[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: TRIX < 0 (momentum turns bearish) OR ADX < 20 (trend weakens)
            if trix[i] < 0 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: TRIX > 0 (momentum turns bullish) OR ADX < 20 (trend weakens)
            if trix[i] > 0 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals