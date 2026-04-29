#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + volume spike + 1d choppiness regime filter
# TRIX(12) = triple smoothed EMA rate of change - captures momentum with less noise
# Long when TRIX crosses above zero AND volume > 1.5x 20-bar average AND 1d chop > 61.8 (ranging)
# Short when TRIX crosses below zero AND volume > 1.5x 20-bar average AND 1d chop > 61.8
# Exit when TRIX crosses zero in opposite direction
# Uses chop regime to avoid whipsaws in strong trends, focus on mean reversion in ranging markets
# Discrete position sizing 0.25 to minimize fee drag. Target: 20-50 trades/year on 4h.

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
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
    
    # Get 1d data for choppiness regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need sufficient data for chop calculation
        return np.zeros(n)
    
    # Calculate 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (highest_high - lowest_low)))
    # Simplified: CHOP = 100 * (ATR(14) sum over 14 bars) / (log10(14) * (max(high) - min(low)))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # Align with index
    
    # ATR(14) on 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index calculation over 14 periods
    atr_sum_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(14) * (highest_high_14 - lowest_low_14)
    chop_1d = 100 * (atr_sum_14 / chop_denominator)
    
    # Align 1d chop to 4h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate TRIX(12) on 4h close: triple smoothed EMA, then ROC
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3.pct_change(periods=1))  # Rate of change of triple smoothed EMA
    trix = trix_raw.values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20, 30)  # Need sufficient history for TRIX (3*12), volume MA, chop
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        chop_val = chop_1d_aligned[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        curr_close = close[i]
        
        # Chop regime filter: only trade when market is ranging (CHOP > 61.8)
        in_chop_regime = chop_val > 61.8
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when TRIX crosses above zero AND volume confirmation AND chop regime
            if trix_prev <= 0 and trix_now > 0 and vol_conf and in_chop_regime:
                signals[i] = 0.25
                position = 1
            # Short when TRIX crosses below zero AND volume confirmation AND chop regime
            elif trix_prev >= 0 and trix_now < 0 and vol_conf and in_chop_regime:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when TRIX crosses below zero
            if trix_prev > 0 and trix_now <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when TRIX crosses above zero
            if trix_prev < 0 and trix_now >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals