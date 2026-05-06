#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction filter with 1w EMA40 trend and volume confirmation
# Uses 1d KAMA (adaptive moving average) for trend identification, 1w EMA40 for higher timeframe trend alignment
# Volume spike (>2.0x 20-bar average) confirms momentum
# Discrete sizing 0.25 to balance profit and fee drag; target 40-80 total trades over 4 years (10-20/year)
# Works in bull/bear: KAMA adapts to volatility, EMA40 filter prevents counter-trend trades, volume ensures participation

name = "1d_KAMA_1wEMA40_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA40 trend filter ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema40_1w = close_1w_series.ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Calculate 1d KAMA (adaptive moving average)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # will be corrected below
    
    # Proper ER calculation
    price_diff = np.diff(close, n=10)
    change_10 = np.abs(price_diff)
    volatility_10 = np.zeros_like(close)
    for i in range(10, len(close)):
        volatility_10[i] = np.sum(np.abs(np.diff(close[i-10:i+1])))
    
    # Avoid division by zero
    er = np.zeros_like(close)
    mask = volatility_10 != 0
    er[mask] = change_10[mask] / volatility_10[mask]
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # Calculate KAMA
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate ATR(14) for stoploss
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume spike filter (>2.0x 20-bar average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Align HTF indicators to 1d timeframe
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)  # Self-align for 1d
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(ema40_1w_aligned[i]) or np.isnan(kama_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                long_extreme = 0.0
                short_extreme = 0.0
            continue
        
        if position == 0:
            # Long: price > KAMA AND uptrend (price > EMA40_1w) AND volume spike
            if close[i] > kama_aligned[i] and close[i] > ema40_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            # Short: price < KAMA AND downtrend (price < EMA40_1w) AND volume spike
            elif close[i] < kama_aligned[i] and close[i] < ema40_1w_aligned[i] and volume_filter[i]:
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