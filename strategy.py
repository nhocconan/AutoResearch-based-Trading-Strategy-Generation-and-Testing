#!/usr/bin/env python3
"""
Hypothesis: 4h TRIX(9) zero-line cross with volume spike and 1d choppiness regime filter.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d Choppiness Index (CHOP) > 61.8 for ranging market (mean reversion), CHOP < 38.2 for trending.
- TRIX: 1-period ROC of triple-smoothed EMA(9). Long when TRIX crosses above zero, short when below zero.
- Volume confirmation: current volume > 2.0 * 20-period volume MA to filter weak breakouts.
- ATR-based stoploss: exit when price moves against position by 2.0 * ATR(14).
- Signal size: 0.25 discrete to balance return and drawdown control.
Designed to capture momentum shifts in both trending and ranging markets with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Choppiness Index (CHOP) = 100 * log10(sum(ATR(14)) / (log10(n) * max(high-low) over n))
    # Using standard CHOP formula: 100 * LOG10( SUM(ATR1,14) / (LOG10(N) * (HHV(HIGH,14) - LLV(LOW,14))) )
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate TR and ATR(14) for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate CHOP(14) for 1d
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denominator = np.log10(14) * (hh - ll)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)  # avoid div by zero
    chop = 100 * np.log10(atr_sum / chop_denominator)
    
    # Align HTF CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX(9) on 4h
    # TRIX = 100 * (EMA(EMA(EMA(close,9),9),9) - prev) / prev
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = 0  # first value undefined
    
    # Calculate ATR(14) for stoploss on 4h
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(100, 34, 20, 14, 20)  # Need enough bars for TRIX, ATR, volume MA, CHOP
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Determine 1d regime: CHOP > 61.8 = ranging (mean reversion), CHOP < 38.2 = trending
            chop_val = chop_aligned[i]
            is_ranging = chop_val > 61.8
            is_trending = chop_val < 38.2
            
            # TRIX zero-line cross signals
            trix_cross_up = trix[i] > 0 and trix[i-1] <= 0
            trix_cross_down = trix[i] < 0 and trix[i-1] >= 0
            
            # Long: TRIX crosses above zero AND volume confirmed
            if trix_cross_up and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: TRIX crosses below zero AND volume confirmed
            elif trix_cross_down and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Long position: exit on stoploss or TRIX cross down
            stop_loss = entry_price - 2.0 * atr[i]
            if curr_low < stop_loss or (trix[i] < 0 and trix[i-1] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on stoploss or TRIX cross up
            stop_loss = entry_price + 2.0 * atr[i]
            if curr_high > stop_loss or (trix[i] > 0 and trix[i-1] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX9_ZeroCross_VolumeSpike_1dChop_Regime_v1"
timeframe = "4h"
leverage = 1.0