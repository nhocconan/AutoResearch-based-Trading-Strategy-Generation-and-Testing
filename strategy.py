#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_trix_volume_regime"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # TRIX calculation (15-period)
    close_1w = df_1w['close'].values
    ema1 = pd.Series(close_1w).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix_raw = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix_raw[0] = np.nan
    trix = pd.Series(trix_raw).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Align TRIX to daily
    trix_aligned = align_htf_to_ltf(prices, df_1w, trix)
    
    # Volume filter: volume > 1.8x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Chop regime filter (daily) - avoid trending markets
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    range_14 = max_high - min_low
    chop = 100 * np.log10(atr.sum() / range_14) / np.log10(14) if isinstance(atr, float) else \
           100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum() / range_14) / np.log10(14)
    chop = pd.Series(chop).fillna(50).values  # Default to neutral when undefined
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(trix_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(chop[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        chop_value = chop[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Chop regime: only trade in choppy markets (CHOP > 50)
        chop_filter = chop_value > 50
        
        # Long conditions: TRIX crosses above 0 with volume and chop
        long_signal = volume_confirmed and chop_filter and (trix_aligned[i] > 0) and (trix_aligned[i-1] <= 0)
        
        # Short conditions: TRIX crosses below 0 with volume and chop
        short_signal = volume_confirmed and chop_filter and (trix_aligned[i] < 0) and (trix_aligned[i-1] >= 0)
        
        # Exit when TRIX crosses zero in opposite direction
        exit_long = position == 1 and (trix_aligned[i] < 0)
        exit_short = position == -1 and (trix_aligned[i] > 0)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: TRIX momentum + volume confirmation + chop regime filter on daily timeframe.
# Uses weekly TRIX (15,9) to capture medium-term momentum, aligned to daily.
# Enters long when TRIX crosses above zero with volume confirmation (>1.8x average)
# and chop regime filter (CHOP > 50 indicating ranging market).
# Enters short when TRIX crosses below zero with same conditions.
# Chop filter prevents trading in strong trends where TRIX whipsaws.
# Volume ensures participation from market actors.
# Target: 20-60 trades over 4 years (5-15/year) to minimize fee drag on daily timeframe.
# Works in both bull and bear markets by capturing momentum reversals in ranging conditions.