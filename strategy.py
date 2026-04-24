#!/usr/bin/env python3
"""
Hypothesis: 1d KAMA trend with 1w volume confirmation and choppiness regime filter.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w for volume average and choppiness calculation.
- KAMA: adapts to market noise, trending in low noise, flat in high noise.
- Entry: Long when price > KAMA AND volume > 1.5 * 1w average volume AND choppiness < 50 (trending regime).
         Short when price < KAMA AND volume > 1.5 * 1w average volume AND choppiness < 50.
- Exit: Opposite KAMA crossover.
- Signal size: 0.25 discrete to minimize fee drag.
- KAMA captures trend direction with lag reduction.
- Volume confirmation ensures institutional participation.
- Choppiness filter avoids ranging markets where trend signals fail.
- Works in both bull and bear markets as it adapts to trend strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average with proper min_periods."""
    close_series = pd.Series(close)
    direction = abs(close_series - close_series.shift(er_period))
    volatility = close_series.diff().abs().rolling(window=er_period, min_periods=1).sum()
    er = direction / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
    kama_values = [close_series.iloc[0]]  # seed
    for i in range(1, len(close_series)):
        kama_values.append(kama_values[-1] + sc.iloc[i] * (close_series.iloc[i] - kama_values[-1]))
    return np.array(kama_values)

def choppiness(high, low, close, period=14):
    """Calculate Choppiness Index with proper min_periods."""
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    atr1 = high_series - low_series
    atr2 = abs(high_series - close_series.shift(1))
    atr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([atr1, atr2, atr3], axis=1).max(axis=1)
    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    hh = high_series.rolling(window=period, min_periods=period).max()
    ll = low_series.rolling(window=period, min_periods=period).min()
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan).fillna(50)
    return chop.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need sufficient data for calculations
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:  # Need sufficient data for volume MA
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20)
    
    # Calculate 1w ATR for choppiness calculation
    if len(df_1w) < 14:  # Need sufficient data for ATR(14)
        return np.zeros(n)
    
    atr_1w = pd.Series(df_1w['high'].values - df_1w['low'].values).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 1w high/low for choppiness calculation
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    chop_1w = choppiness(high_1w, low_1w, close_1w, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate KAMA from 1d data
    kama_vals = kama(close, 10, 2, 30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 30  # Need 30 for KAMA seed
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(vol_ma_20_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(chop_1w_aligned[i]) or np.isnan(kama_vals[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        prev_close = close[i-1]
        prev_kama = kama_vals[i-1]
        
        # Exit conditions: opposite KAMA crossover
        if position != 0:
            # Exit long: price crosses below KAMA
            if position == 1:
                if curr_close <= kama_vals[i] and prev_close > prev_kama:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price crosses above KAMA
            elif position == -1:
                if curr_close >= kama_vals[i] and prev_close < prev_kama:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: KAMA crossover with volume confirmation and choppiness filter
        if position == 0:
            # KAMA crossover signals
            cross_up = curr_close > kama_vals[i] and prev_close <= prev_kama
            cross_down = curr_close < kama_vals[i] and prev_close >= prev_kama
            
            # Volume confirmation: current volume > 1.5 * 20-period average volume (aligned)
            volume_confirm = curr_volume > 1.5 * vol_ma_20_aligned[i] if not np.isnan(vol_ma_20_aligned[i]) else False
            
            # Choppiness filter: chop < 50 (trending regime)
            chop_filter = chop_1w_aligned[i] < 50
            
            if cross_up and volume_confirm and chop_filter:
                signals[i] = 0.25
                position = 1
            elif cross_down and volume_confirm and chop_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "1d_KAMA10_1wVolumeSpike_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0