#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX(9) + volume spike + 1d choppiness regime filter
# TRIX(9) catches momentum reversals with low lag
# Volume confirmation > 2.0x 20-period EMA ensures institutional participation
# 1d Choppiness Index > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
# In ranging markets (CHOP > 61.8): fade TRIX extremes (long when TRIX crosses below -0.1, short when above +0.1)
# In trending markets (CHOP < 38.2): follow TRIX momentum (long when TRIX crosses above zero, short when below zero)
# Designed for low trade frequency: ~12-30 trades/year per symbol with 0.25 sizing
# Works in both bull and bear markets by adapting to regime

name = "12h_TRIX9_Volume_Chopper_v1"
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
    
    # 1d HTF data for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    # Calculate 1d True Range for Choppiness Index
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).values
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d Choppiness Index: CHOP = 100 * LOG10(SUM(ATR14,14) / (MAXHIGH14 - MINLOW14)) / LOG10(14)
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid div by zero
    chop_raw = 100 * np.log10(sum_atr_14 / chop_denom) / np.log10(14)
    chop_1d = pd.Series(chop_raw).ewm(span=1, adjust=False).mean().values  # smooth
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate TRIX(9) on 12h close
    # TRIX = EMA(EMA(EMA(close, 9), 9), 9) - 1, then * 100 for percentage
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = (ema3 / np.roll(ema3, 1) - 1) * 100
    trix[0] = 0  # first value has no previous
    
    # Volume confirmation: volume > 2.0 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d data for CHOP (15 days) + TRIX needs 9*3=27 bars for stability
    start_idx = 40
    
    for i in range(start_idx, n):
        if (np.isnan(chop_1d_aligned[i]) or np.isnan(trix[i]) or 
            np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        chop = chop_1d_aligned[i]
        trix_val = trix[i]
        trix_prev = trix[i-1] if i > 0 else 0
        
        # Regime detection: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
        is_ranging = chop > 61.8
        is_trending = chop < 38.2
        
        if position == 0:  # Flat - look for new entries
            if is_ranging:
                # In ranging markets: fade TRIX extremes
                if trix_prev >= -0.1 and trix_val < -0.1 and volume_spike[i]:
                    # TRIX crosses below -0.1 -> long (mean reversion up)
                    signals[i] = 0.25
                    position = 1
                elif trix_prev <= 0.1 and trix_val > 0.1 and volume_spike[i]:
                    # TRIX crosses above +0.1 -> short (mean reversion down)
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            elif is_trending:
                # In trending markets: follow TRIX momentum
                if trix_prev <= 0 and trix_val > 0 and volume_spike[i]:
                    # TRIX crosses above zero -> long (momentum up)
                    signals[i] = 0.25
                    position = 1
                elif trix_prev >= 0 and trix_val < 0 and volume_spike[i]:
                    # TRIX crosses below zero -> short (momentum down)
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                # Choppy transition zone (38.2 <= CHOP <= 61.8): no trades
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: opposite TRIX cross
            if is_ranging and trix_prev <= -0.1 and trix_val >= -0.1:
                # TRIX crosses back above -0.1 in ranging market
                signals[i] = 0.0
                position = 0
            elif is_trending and trix_prev >= 0 and trix_val <= 0:
                # TRIX crosses back below zero in trending market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: opposite TRIX cross
            if is_ranging and trix_prev >= 0.1 and trix_val <= 0.1:
                # TRIX crosses back below +0.1 in ranging market
                signals[i] = 0.0
                position = 0
            elif is_trending and trix_prev <= 0 and trix_val >= 0:
                # TRIX crosses back above zero in trending market
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals