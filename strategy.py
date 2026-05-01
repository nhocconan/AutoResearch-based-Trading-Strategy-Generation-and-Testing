#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum with 1d volume spike regime and choppiness filter.
# TRIX (12) captures smoothed momentum; long when TRIX crosses above zero AND 1d volume > 1.5x 20-day average AND chop < 61.8 (trending regime).
# Short when TRIX crosses below zero AND 1d volume > 1.5x 20-day average AND chop < 61.8.
# Uses discrete sizing 0.25. Target: 75-200 total trades over 4 years.
# Volume spike confirms institutional participation; chop filter avoids whipsaws in ranging markets.
# Works in bull markets (momentum continuation) and bear markets (mean reversion at extremes via zero-cross).

name = "4h_TRIX_ZeroCross_1dVolumeSpike_ChopFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume and chop filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for 1d indicators
        return np.zeros(n)
    
    # 1d volume: current volume > 1.5x 20-bar average
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, vol_1d > (vol_ma_1d * 1.5))
    
    # 1d choppiness index: CHOP(14) < 61.8 = trending regime (avoid ranging markets)
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    true_range_1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                               np.maximum(np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                          np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    true_range_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first bar
    atr_14_1d = pd.Series(true_range_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denominator = max_high_14 - min_low_14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop_1d = 100 * np.log10(sum_atr_14 / chop_denominator) / np.log10(14)
    chop_filter_1d = align_htf_to_ltf(prices, df_1d, chop_1d < 61.8)
    
    # 4h TRIX(12,9,9): triple EMA of ROC, then signal line
    # TRIX = EMA(EMA(EMA(ROC, 12), 9), 9)
    roc = pd.Series(close).pct_change(periods=12).values  # ROC(12)
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = ema3  # TRIX value
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values  # signal line
    trix_hist = trix - trix_signal  # histogram for zero-cross
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # warmup for TRIX and 1d indicators
    
    for i in range(start_idx, n):
        if np.isnan(trix_hist[i]) or np.isnan(vol_spike_1d[i]) or np.isnan(chop_filter_1d[i]):
            signals[i] = 0.0
            continue
        
        # TRIX zero-cross signals
        trix_cross_up = trix_hist[i] > 0 and trix_hist[i-1] <= 0  # cross above zero
        trix_cross_down = trix_hist[i] < 0 and trix_hist[i-1] >= 0  # cross below zero
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: TRIX crosses above zero AND 1d volume spike AND chop < 61.8 (trending)
            if (trix_cross_up and 
                vol_spike_1d[i] and 
                chop_filter_1d[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND 1d volume spike AND chop < 61.8 (trending)
            elif (trix_cross_down and 
                  vol_spike_1d[i] and 
                  chop_filter_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: TRIX crosses below zero (momentum loss) OR chop >= 61.8 (ranging) OR volume spike ends
            if (trix_hist[i] < 0 or 
                chop_filter_1d[i] == False or 
                vol_spike_1d[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero (momentum loss) OR chop >= 61.8 (ranging) OR volume spike ends
            if (trix_hist[i] > 0 or 
                chop_filter_1d[i] == False or 
                vol_spike_1d[i] == False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals