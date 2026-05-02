#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX crossover with 1d volume spike and choppiness regime filter
# Uses TRIX(12) on 12h for momentum signal, confirmed by 1d volume > 1.5x 20-period average
# Only takes signals when 1d choppiness index > 61.8 (range regime) to avoid whipsaws in strong trends
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by using regime filter to adapt to market conditions
# TRIX is effective at catching reversals in ranging markets which dominate bear markets like 2025+

name = "12h_TRIX_VolumeChop_v1"
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
    
    # Load 1d data ONCE before loop for volume and choppiness filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d volume confirmation (1.5x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm_1d = df_1d['volume'].values > (vol_ma_1d * 1.5)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    # Calculate 1d choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / (n-1))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    tr_1d = np.maximum(
        df_1d['high'].values - df_1d['low'].values,
        np.maximum(
            np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
            np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        )
    )
    tr_1d[0] = df_1d['high'].values[0] - df_1d['low'].values[0]  # first TR
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14_1d).rolling(window=14, min_periods=14).sum().values
    hh_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(hh_14 - ll_14) * 14
    chop_numerator = np.log10(sum_atr_14) * 14
    chop_1d = 100 - (100 * chop_numerator / chop_denominator)
    chop_1d[~np.isfinite(chop_1d)] = 50  # set neutral when invalid
    chop_regime = chop_1d > 61.8  # ranging regime
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    # Calculate 12h TRIX (12-period triple EMA)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean()
    trix = (ema3 - pd.Series(ema3).shift(1)) / pd.Series(ema3).shift(1) * 100
    trix_values = trix.values
    trix_signal = pd.Series(trix_values).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for TRIX and 1d indicators)
    start_idx = 50  # max(12*3 for TRIX, 20 for volume, 14 for CHOP) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(trix_signal[i]) or np.isnan(trix_signal[i-1]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # TRIX crossover signals
        trix_cross_up = trix_signal[i-1] <= trix_signal[i] and trix_signal[i] > 0
        trix_cross_down = trix_signal[i-1] >= trix_signal[i] and trix_signal[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long: TRIX crosses above zero AND volume confirm AND ranging regime
            if (trix_cross_up and 
                volume_confirm_1d_aligned[i] and 
                chop_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero AND volume confirm AND ranging regime
            elif (trix_cross_down and 
                  volume_confirm_1d_aligned[i] and 
                  chop_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: TRIX crosses below zero
            if trix_signal[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero
            if trix_signal[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals