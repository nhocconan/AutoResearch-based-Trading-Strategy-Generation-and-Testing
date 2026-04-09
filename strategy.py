#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA trend with 1w volume confirmation and chop regime filter
# - Uses KAMA(10) on 1d for trend direction (long when price > KAMA, short when price < KAMA)
# - Confirms with 1w volume > 1.8x 10-period average (institutional participation)
# - Filters by 1d choppiness index: only trade when CHOP > 61.8 (range) or CHOP < 38.2 (trend)
# - Exits when price crosses KAMA in opposite direction
# - Position size: 0.25 (25% of capital) to balance return and drawdown
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to minimize fee drag
# - Works in bull markets (trends continue) and bear markets (mean reversion in range)

name = "1d_kama_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # 1d KAMA(10) - Kaufman Adaptive Moving Average
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        if i >= 10:
            change_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            volatility_sum = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
            if volatility_sum > 0:
                er[i] = change_sum / volatility_sum
            else:
                er[i] = 0
    er[:10] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[:10] = close_1d[:10]
    for i in range(10, n):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # 1d ATR(14) for stoploss reference
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1d Choppiness Index(14)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = np.where((highest_14 - lowest_14) > 0, highest_14 - lowest_14, 1e-10)
    chop = 100 * np.log10(sum_tr_14 / chop_denom) / np.log10(14)
    chop_range = chop > 61.8  # range-bound market
    chop_trend = chop < 38.2  # trending market
    
    # Pre-compute 1w indicators
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # 1w Volume > 1.8x 10-period average
    avg_volume_10 = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
    volume_spike_1w = volume_1w > (1.8 * avg_volume_10)
    
    # Align all indicators to 1d
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_1d}), kama)
    volume_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_spike_1w.astype(float))
    chop_range_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_1d}), chop_range.astype(float))
    chop_trend_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_1d}), chop_trend.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close_1d}), atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_aligned[i]) or np.isnan(volume_spike_1w_aligned[i]) or
            np.isnan(chop_range_aligned[i]) or np.isnan(chop_trend_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit when price crosses below KAMA
            if close_1d[i] <= kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price crosses above KAMA
            if close_1d[i] >= kama_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for KAMA alignment with volume confirmation and regime filter
            if (close_1d[i] > kama_aligned[i] and  # Price above KAMA (bullish)
                volume_spike_1w_aligned[i] and    # Weekly volume confirmation
                (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = 1
                signals[i] = 0.25
            elif (close_1d[i] < kama_aligned[i] and   # Price below KAMA (bearish)
                  volume_spike_1w_aligned[i] and     # Weekly volume confirmation
                  (chop_range_aligned[i] or chop_trend_aligned[i])):  # Either regime
                position = -1
                signals[i] = -0.25
    
    return signals