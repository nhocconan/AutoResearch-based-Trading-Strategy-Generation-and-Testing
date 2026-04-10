#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX momentum + 1d volume spike + choppiness regime filter
# - Long when TRIX crosses above zero on 4h, 1d volume > 1.5x 20-bar average, and CHOP > 61.8 (range regime)
# - Short when TRIX crosses below zero on 4h, 1d volume spike, and CHOP > 61.8
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or TRIX reverses
# - Targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - TRIX catches momentum shifts in bear market rallies; volume confirms participation; chop filter avoids whipsaws in strong trends

name = "4h_1d_trix_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d TRIX (15,9,9) - triple EMA
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix_raw = 100 * (ema3 / ema3.shift(1) - 1)
    trix = trix_raw.values
    trix_prev = np.roll(trix, 1)
    trix_prev[0] = np.nan
    trix_zero_cross_up = (trix > 0) & (trix_prev <= 0)
    trix_zero_cross_down = (trix < 0) & (trix_prev >= 0)
    trix_zero_cross_up_aligned = align_htf_to_ltf(prices, df_1d, trix_zero_cross_up)
    trix_zero_cross_down_aligned = align_htf_to_ltf(prices, df_1d, trix_zero_cross_down)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d Choppiness Index (CHOP) regime filter: CHOP > 61.8 = ranging (good for mean reversion in bear rallies)
    # CHOP = 100 * log10(sum(ATR(14)) / (n * log(n))) / log10(n)
    atr_14_chop = np.zeros_like(tr)
    atr_14_chop[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_chop[i] = (atr_14_chop[i-1] * (14-1) + tr[i]) / 14
    
    sum_atr = np.zeros_like(atr_14_chop)
    for i in range(14, len(atr_14_chop)):
        sum_atr[i] = np.sum(atr_14_chop[i-13:i+1])  # 14-period sum
    
    n_val = 14
    chop = np.zeros_like(close_1d)
    for i in range(n_val, len(close_1d)):
        if sum_atr[i] > 0 and close_1d[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / (n_val * np.log10(n_val))) / np.log10(n_val)
        else:
            chop[i] = 50.0  # neutral
    chop_regime = chop > 61.8  # ranging regime
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(trix_zero_cross_up_aligned[i]) or np.isnan(trix_zero_cross_down_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or TRIX reverses (momentum loss)
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                trix_zero_cross_down_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or TRIX reverses (momentum loss)
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                trix_zero_cross_up_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for TRIX zero cross with volume spike and chop regime
            if vol_spike_1d_aligned[i] and chop_regime_aligned[i]:
                # Long signal: TRIX crosses above zero
                if trix_zero_cross_up_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: TRIX crosses below zero
                elif trix_zero_cross_down_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals